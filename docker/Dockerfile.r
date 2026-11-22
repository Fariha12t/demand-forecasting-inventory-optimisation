FROM rocker/r-ver:4.4.1
RUN install2.r --error --skipinstalled forecast ggplot2 jsonlite data.table tseries zoo
WORKDIR /app
COPY r ./r
CMD ["Rscript", "r/02_forecast_models.R"]
