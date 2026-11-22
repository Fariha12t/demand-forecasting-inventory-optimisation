"""Optional temporal deep-learning model: a global LSTM over all series (PyTorch).

Disabled by default (`models.lstm.enabled: false`) because gradient boosting with lag features is
usually as accurate on retail count data at a fraction of the cost - enable it to *test* whether a
sequence model is justified on your data. Requires `pip install -r requirements-lstm.txt`.

Design: input window of the last W days (scaled sales + calendar/price/promo covariates), the target
day's known covariates and a learned series embedding -> next-day scaled sales. Multi-step forecasts are
produced recursively, like the tree models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def build_panel(df: pd.DataFrame, price_ref_days: int = 56):
    """Return ids, dates, sales (T,N), covariates (T,N,7) from a balanced (date, id) panel."""
    df = df.sort_values(["date", "id"]).reset_index(drop=True)
    ids = np.sort(df["id"].unique())
    dates = np.sort(df["date"].unique())
    T, N = len(dates), len(ids)
    shape = (T, N)
    sales = df["sales"].to_numpy(float).reshape(shape)
    price = pd.DataFrame(df["sell_price"].to_numpy(float).reshape(shape))
    price_rel = (price / price.rolling(price_ref_days, min_periods=1).max()).to_numpy()
    promo = df["promo"].to_numpy(float).reshape(shape)
    event = df["is_event"].to_numpy(float).reshape(shape)
    d = pd.DatetimeIndex(dates)
    dow, doy = d.dayofweek.to_numpy(), d.dayofyear.to_numpy()
    cal = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], axis=1)  # (T,4)
    cov = np.concatenate([np.repeat(cal[:, None, :], N, axis=1), price_rel[..., None],
                          promo[..., None], event[..., None]], axis=2)  # (T,N,7)
    return ids, dates, sales, cov


def make_windows(scaled: np.ndarray, cov: np.ndarray, window: int, stride: int = 1):
    """Training windows: (seq (M,W,8), target-day covariates (M,7), series idx (M,), y (M,))."""
    T, N = scaled.shape
    seqs, covs, sids, ys = [], [], [], []
    for t in range(window, T, stride):
        x = np.concatenate([scaled[t - window:t].T[..., None], cov[t - window:t].transpose(1, 0, 2)], axis=2)
        seqs.append(x)
        covs.append(cov[t])
        sids.append(np.arange(N))
        ys.append(scaled[t])
    return (np.concatenate(seqs).astype("float32"), np.concatenate(covs).astype("float32"),
            np.concatenate(sids).astype("int64"), np.concatenate(ys).astype("float32"))


class LSTMForecaster:
    def __init__(self, cfg: dict):
        self.c = cfg["models"]["lstm"]
        self.price_ref_days = cfg["features"]["price_ref_days"]
        self.seed = cfg["project"]["seed"]

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.ids, _, sales, cov = build_panel(train, self.price_ref_days)
        self.scale = sales.mean(axis=0) + 1e-3
        seq, cv, sid, y = make_windows(sales / self.scale, cov, self.c["window"], self.c["stride"])
        N = len(self.ids)

        class Net(nn.Module):
            def __init__(self, hidden=48, emb=8):
                super().__init__()
                self.lstm = nn.LSTM(8, hidden, batch_first=True)
                self.emb = nn.Embedding(N, emb)
                self.head = nn.Sequential(nn.Linear(hidden + 7 + emb, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, s, c, i):
                _, (h, _) = self.lstm(s)
                z = torch.cat([h[-1], c, self.emb(i)], dim=1)
                return torch.nn.functional.softplus(self.head(z)).squeeze(1)

        self.net = Net(self.c["hidden"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["lr"])
        tens = [torch.from_numpy(a) for a in (seq, cv, sid, y)]
        n, bs = len(y), self.c["batch_size"]
        self.net.train()
        for _ in range(self.c["epochs"]):
            perm = torch.randperm(n)
            for k in range(0, n, bs):
                b = perm[k:k + bs]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(tens[0][b], tens[1][b], tens[2][b]), tens[3][b])
                loss.backward()
                opt.step()
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        import torch

        fut = future.copy()
        fut["sales"] = np.nan
        comb = pd.concat([history[fut.columns.intersection(history.columns)], fut], ignore_index=True)
        ids, dates, sales, cov = build_panel(comb.assign(sales=comb["sales"].fillna(0.0)), self.price_ref_days)
        assert (ids == self.ids).all(), "series set changed since fit"
        scaled = sales / self.scale
        n_hist = history["date"].nunique()
        W = self.c["window"]
        self.net.eval()
        with torch.no_grad():
            for t in range(n_hist, len(dates)):
                x = np.concatenate([scaled[t - W:t].T[..., None], cov[t - W:t].transpose(1, 0, 2)], axis=2)
                out = self.net(torch.from_numpy(x.astype("float32")), torch.from_numpy(cov[t].astype("float32")),
                               torch.arange(len(ids))).numpy()
                scaled[t] = out
        pred = np.clip(scaled[n_hist:] * self.scale, 0, None)
        res = pd.DataFrame({"id": np.tile(ids, len(dates) - n_hist),
                            "date": np.repeat(dates[n_hist:], len(ids)), "y_pred": pred.ravel()})
        return res
