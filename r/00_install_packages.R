# Install the R packages used by the project (run once: `make install-r`).
pkgs <- c("forecast", "ggplot2", "jsonlite", "data.table", "tseries", "zoo")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only = TRUE))
cat("R packages ready:", paste(pkgs, collapse = ", "), "\n")
