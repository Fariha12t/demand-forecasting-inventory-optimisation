# Are the accuracy differences between models statistically meaningful?
# Reads the forecasts written by the Python pipeline (outputs/forecasts_test.csv, which already includes the R models)
# and runs, on per-series MAE over the test window:
#   * Friedman rank-sum test across all models
#   * pairwise Wilcoxon signed-rank tests vs the best model, Holm-adjusted
# Usage:  Rscript r/03_model_significance.R   (after `make train`)
source("r/utils.R")

fc <- fread(file.path(OUT, "forecasts_test.csv"))
mae_tbl <- fc[, .(mae = mean(abs(y_true - y_pred))), by = .(id, model)]
wide <- dcast(mae_tbl, id ~ model, value.var = "mae")
mat <- as.matrix(wide[, -1]); rownames(mat) <- wide$id

fr <- friedman.test(mat)
cat(sprintf("Friedman test: chi-squared = %.1f, df = %d, p = %.3g\n", fr$statistic, fr$parameter, fr$p.value))

mean_rank <- sort(colMeans(t(apply(mat, 1, rank))))
best <- names(mean_rank)[1]
others <- setdiff(colnames(mat), best)
pw <- rbindlist(lapply(others, function(m) {
  w <- suppressWarnings(wilcox.test(mat[, best], mat[, m], paired = TRUE))
  data.table(best = best, vs = m, mean_mae_best = mean(mat[, best]), mean_mae_vs = mean(mat[, m]),
             median_diff = median(mat[, m] - mat[, best]), p_value = w$p.value)
}))
pw[, p_holm := p.adjust(p_value, method = "holm")][, significant_5pct := p_holm < 0.05]
setorder(pw, p_value)
res <- rbind(data.table(best = best, vs = "(Friedman test, all models)", mean_mae_best = NA_real_, mean_mae_vs = NA_real_,
                        median_diff = NA_real_, p_value = fr$p.value, p_holm = fr$p.value, significant_5pct = fr$p.value < 0.05), pw)
fwrite(res, file.path(ROUT, "model_significance.csv"))
fwrite(data.table(model = names(mean_rank), mean_rank = as.numeric(mean_rank)), file.path(ROUT, "model_mean_ranks.csv"))
print(res)

ggsave(file.path(ROUT, "model_rank_plot.png"),
       ggplot(data.table(model = factor(names(mean_rank), levels = rev(names(mean_rank))), rank = mean_rank),
              aes(model, rank)) + geom_col(fill = "#1F4E79") + coord_flip() +
         labs(title = "Mean rank of per-series MAE on the test window (lower is better)", x = NULL, y = "mean rank"),
       width = 8, height = 4.5, dpi = 120)
cat("Best model by mean rank:", best, "\n")
