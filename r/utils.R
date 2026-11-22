# Shared helpers for the R scripts. Run all scripts from the project root.
suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(ggplot2)
})

PROC <- Sys.getenv("DF_PROCESSED", "data/processed")
OUT  <- Sys.getenv("DF_OUTPUTS", "outputs")
ROUT <- file.path(OUT, "r")
dir.create(ROUT, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 12))   # light theme

load_sales <- function() {
  s <- fread(file.path(PROC, "sales.csv"))
  s[, date := as.Date(date)]
  setorder(s, id, date)
  s
}

load_split <- function() fromJSON(file.path(PROC, "split.json"))

# price relative to the trailing 56-day maximum (only discounts move it)
add_price_rel <- function(s, window = 56) {
  s[, ref := frollapply(sell_price, window, max, align = "right"), by = id]
  s[is.na(ref), ref := cummax(sell_price), by = id]
  s[, price_rel := sell_price / ref]
  s
}

# Seasonal / trend strength (Wang, Smith & Hyndman 2006): 0 = none, 1 = perfectly strong
strength <- function(component, remainder) {
  max(0, 1 - var(remainder, na.rm = TRUE) / var(component + remainder, na.rm = TRUE))
}
