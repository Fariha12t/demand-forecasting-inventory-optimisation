# Statistical EDA in R: seasonality, trend, stationarity and price effects.
# Usage (from project root):  Rscript r/01_eda_seasonality.R
source("r/utils.R")
suppressPackageStartupMessages({ library(forecast); library(tseries) })

s <- load_sales()
cat("Series:", uniqueN(s$id), "| days:", uniqueN(s$date), "\n")

# ---- 1. Total demand: multi-seasonal STL decomposition (weekly + yearly) -------------
tot <- s[, .(sales = sum(sales)), by = date]
y <- msts(tot$sales, seasonal.periods = c(7, 365.25), start = c(as.numeric(format(min(tot$date), "%Y")), 1))
dec <- mstl(y, iterate = 3)
trend <- dec[, "Trend"]; rem <- dec[, "Remainder"]
S7 <- dec[, "Seasonal7"]; S365 <- dec[, "Seasonal365.25"]

summary_tbl <- data.table(
  metric = c("trend_strength", "weekly_seasonality_strength", "yearly_seasonality_strength",
             "adf_p_value_level", "adf_p_value_diff", "kpss_p_value_level",
             "ljung_box_p_remainder_lag14"),
  value = c(strength(trend, rem), strength(S7, rem), strength(S365, rem),
            suppressWarnings(adf.test(tot$sales)$p.value),
            suppressWarnings(adf.test(diff(tot$sales))$p.value),
            suppressWarnings(kpss.test(tot$sales)$p.value),
            Box.test(rem, lag = 14, type = "Ljung-Box")$p.value))
fwrite(summary_tbl, file.path(ROUT, "seasonality_summary.csv"))
print(summary_tbl)

png(file.path(ROUT, "stl_decomposition.png"), width = 1100, height = 900, res = 110)
print(autoplot(dec) + ggtitle("MSTL decomposition of total daily demand (weekly + yearly seasonality)"))
dev.off()

png(file.path(ROUT, "acf.png"), width = 1000, height = 500, res = 110)
print(ggAcf(diff(tot$sales), lag.max = 56) + ggtitle("ACF of differenced total demand (peaks at lag 7, 14, ...)"))
dev.off()

# ---- 2. Seasonal indices -----------------------------------------------------------
tot[, `:=`(dow = factor(weekdays(date, abbreviate = TRUE), levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")),
           month = factor(months(date, abbreviate = TRUE), levels = month.abb))]
tot[, trend_ma := as.numeric(ma(sales, 7, centre = FALSE))]
dow_idx <- tot[!is.na(trend_ma), .(index = mean(sales / trend_ma)), by = dow][order(dow)]
mon_idx <- tot[, .(avg = mean(sales)), by = month][order(month)][, index := avg / mean(tot$sales)][, avg := NULL]
fwrite(dow_idx, file.path(ROUT, "weekday_index.csv")); fwrite(mon_idx, file.path(ROUT, "monthly_index.csv"))
ggsave(file.path(ROUT, "weekday_pattern.png"),
       ggplot(dow_idx, aes(dow, index)) + geom_col(fill = "#1F4E79") + geom_hline(yintercept = 1, linetype = 2) +
         labs(title = "Weekday seasonal index (1 = average day)", x = NULL, y = "index"), width = 7, height = 4, dpi = 120)
ggsave(file.path(ROUT, "monthly_pattern.png"),
       ggplot(mon_idx, aes(month, index, group = 1)) + geom_line(colour = "#1F4E79", linewidth = 1) + geom_point() +
         geom_hline(yintercept = 1, linetype = 2) + labs(title = "Monthly seasonal index", x = NULL, y = "index"),
       width = 7, height = 4, dpi = 120)

# ---- 3. Seasonality by department ---------------------------------------------------
dept <- s[, .(sales = sum(sales)), by = .(dept_id, date)]
dept_tbl <- rbindlist(lapply(split(dept, by = "dept_id"), function(d) {
  setorder(d, date)
  m <- mstl(msts(d$sales, seasonal.periods = c(7, 365.25)), iterate = 3)
  data.table(dept_id = d$dept_id[1], weekly_strength = strength(m[, "Seasonal7"], m[, "Remainder"]),
             yearly_strength = strength(m[, "Seasonal365.25"], m[, "Remainder"]),
             trend_strength = strength(m[, "Trend"], m[, "Remainder"]))
}))
fwrite(dept_tbl, file.path(ROUT, "seasonality_by_dept.csv")); print(dept_tbl)

# ---- 4. Price effects: log-linear regression with series fixed effects (cf. Python GLM) -----
s <- add_price_rel(s)
s[, `:=`(dow = factor(weekdays(date)), month = factor(months(date)), logp = log(price_rel), lsales = log1p(sales))]
el <- rbindlist(lapply(split(s, by = "dept_id"), function(d) {
  fit <- lm(lsales ~ logp + promo + dow + month + id, data = d)
  ci <- confint(fit)["logp", ]
  data.table(dept_id = d$dept_id[1], elasticity_loglinear = coef(fit)["logp"], ci_low = ci[1], ci_high = ci[2],
             promo_uplift_pct = 100 * (exp(coef(fit)["promo"]) - 1), r_squared = summary(fit)$r.squared)
}))
fwrite(el, file.path(ROUT, "price_elasticity_r.csv")); print(el)
ggsave(file.path(ROUT, "price_vs_demand.png"),
       ggplot(s[, .(rel_sales = mean(sales / mean(sales)), n = .N), by = .(dept_id, disc = round((1 - price_rel) * 20) / 20)][n > 200],
              aes(disc, rel_sales, colour = dept_id)) + geom_line() + geom_point() +
         labs(title = "Relative demand vs discount depth", x = "discount (share of regular price)", y = "sales / series mean"),
       width = 8, height = 4.5, dpi = 120)
cat("EDA outputs written to", ROUT, "\n")
