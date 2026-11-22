# Statistical forecasting in R: ETS, seasonal ARIMA and ARIMAX with Fourier terms (dynamic harmonic regression).
# Produces forecasts on exactly the same time-based folds as the Python pipeline:
#   fold "val":  fit on data <= train_end,  forecast the validation window
#   fold "test": fit on data <= val_end,    forecast the test window
# Output: outputs/r/r_forecasts.csv  (id, date, fold, model, y_pred) - merged into the model comparison by `make train`.
#
# Usage:  Rscript r/02_forecast_models.R           (env: SERIES_LIMIT=5 for a quick run, R_CORES=4)
source("r/utils.R")
suppressPackageStartupMessages({ library(forecast); library(parallel) })

TRAIN_DAYS <- 730            # fit window (recent history)
K <- c(3, 4)                 # Fourier terms: weekly (max 3), yearly

s <- add_price_rel(load_sales())
split <- load_split()
ids <- sort(unique(s$id))
limit <- as.integer(Sys.getenv("SERIES_LIMIT", "0"))
if (limit > 0) ids <- ids[seq_len(limit)]
cores <- if (.Platform$OS.type == "windows") 1L else as.integer(Sys.getenv("R_CORES", max(1, detectCores() - 1)))
cat("Fitting", length(ids), "series on", cores, "core(s)\n")

folds <- list(val  = list(origin = split$train_end, from = split$val_start,  to = split$val_end),
              test = list(origin = split$val_end,   from = split$test_start, to = split$test_end))

safe <- function(expr, h) {
  out <- tryCatch(as.numeric(expr), error = function(e) rep(NA_real_, h))
  pmax(out, 0)
}

fit_series <- function(sid, fold, f) {
  d <- s[id == sid]
  train <- tail(d[date <= as.Date(f$origin)], TRAIN_DAYS)
  fut <- d[date >= as.Date(f$from) & date <= as.Date(f$to)]
  h <- nrow(fut)
  y7 <- ts(train$sales, frequency = 7)
  ym <- msts(train$sales, seasonal.periods = c(7, 365.25))
  fallback <- rep(mean(tail(train$sales, 28)), h)

  p_ets <- safe(forecast(ets(y7), h = h)$mean, h)
  p_arima <- safe(forecast(auto.arima(y7, stepwise = TRUE, approximation = TRUE), h = h)$mean, h)
  p_dhr <- safe({
    xr <- cbind(fourier(ym, K = K), price_rel = train$price_rel, promo = train$promo)
    xf <- cbind(fourier(ym, K = K, h = h), price_rel = fut$price_rel, promo = fut$promo)
    fit <- auto.arima(ym, xreg = xr, seasonal = FALSE, stepwise = TRUE, approximation = TRUE)
    forecast(fit, xreg = xf, h = h)$mean
  }, h)
  fix <- function(p) ifelse(is.finite(p), p, fallback)

  rbindlist(list(
    data.table(id = sid, date = fut$date, fold = fold, model = "R_ETS", y_pred = fix(p_ets)),
    data.table(id = sid, date = fut$date, fold = fold, model = "R_ARIMA", y_pred = fix(p_arima)),
    data.table(id = sid, date = fut$date, fold = fold, model = "R_ARIMAX_Fourier", y_pred = fix(p_dhr))))
}

t0 <- Sys.time()
res <- rbindlist(lapply(names(folds), function(fold) {
  cat("Fold", fold, "...\n")
  rbindlist(mclapply(ids, fit_series, fold = fold, f = folds[[fold]], mc.cores = cores))
}))
fwrite(res, file.path(ROUT, "r_forecasts.csv"))
cat(sprintf("Wrote %d forecasts (%s) in %.1f min\n", nrow(res), paste(unique(res$model), collapse = ", "),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
