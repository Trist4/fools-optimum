import pandas as pd
import numpy as np
import time

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

##############################################################################
# SETTINGS
##############################################################################

print('\nStarting')

CSV_PATH = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/master_df.csv"

FORECAST_HORIZON = 4
EXCLUDE_LAST = 0

INITIAL_TRAIN_RATIO = 0.80

ROLLING_STEP = FORECAST_HORIZON
1
# ARIMA(p, d, q)
ARIMA_ORDER = (2, 1, 5)

##############################################################################
# READ DATA
##############################################################################

df = pd.read_csv(CSV_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

##############################################################################
# PRICE COMPONENTS
##############################################################################

price_components = [
    "Fuel_Tax",
    "Customs_Excise",
    "Equalization_Fund_Levy",
    "Road_Accident_Fund",
    "Transport_Cost",
    "Petroleum_Products_Levy",
    "Wholesale_Margin",
    "Secondary_Storage",
    "Secondary_Distribution",
    "Retail_Margin",
    "Slate_Levy",
    "Delivery_Cost",
    "DSML",
    "Incremental_Inland_Transport_Recovery_Cost"
]

##############################################################################
# CREATE TOTAL RETAIL PRICE
##############################################################################

df["Taxes"] = df[price_components].sum(axis=1)

df["Retail_Price"] = (
    df["BFP"] +
    df["Taxes"]
)

##############################################################################
# REMOVE MISSING VALUES
##############################################################################

df = df.dropna(
    subset=["Retail_Price"]
).reset_index(drop=True)

##############################################################################
# INITIAL TRAIN SIZE
##############################################################################

initial_train_size = int(
    len(df) * INITIAL_TRAIN_RATIO
)

##############################################################################
# STORAGE
##############################################################################

window_metrics = []

prediction_table = []

##############################################################################
# ROLLING ORIGIN VALIDATION
##############################################################################

window = 1

for train_end in range(
        initial_train_size,
        len(df) - FORECAST_HORIZON - EXCLUDE_LAST + 1,
        ROLLING_STEP):

    tic = time.time()

    ##########################################################################
    # TRAINING DATA
    ##########################################################################

    y_train = df.iloc[
        :train_end
    ]["Retail_Price"]

    ##########################################################################
    # FIT ARIMA
    ##########################################################################

    model = ARIMA(
        y_train,
        order=ARIMA_ORDER
    )

    fitted_model = model.fit()

    ##########################################################################
    # FORECAST
    ##########################################################################

    predictions = fitted_model.forecast(
        steps=FORECAST_HORIZON
    )

    predictions = np.asarray(
        predictions
    )

    ##########################################################################
    # ACTUAL FUTURE RETAIL PRICE
    ##########################################################################

    actual_retail = df.iloc[
        train_end:
        train_end + FORECAST_HORIZON
    ]["Retail_Price"].values

    ##########################################################################
    # FORECAST DATES
    ##########################################################################

    dates = df.iloc[
        train_end:
        train_end + FORECAST_HORIZON
    ]["Date"]

    toc = time.time()

    ##########################################################################
    # METRICS
    ##########################################################################

    rmse = np.sqrt(
        mean_squared_error(
            actual_retail,
            predictions
        )
    )

    mae = mean_absolute_error(
        actual_retail,
        predictions
    )

    mape = mean_absolute_percentage_error(
        actual_retail,
        predictions
    )

    if len(actual_retail) > 1:

        r2 = r2_score(
            actual_retail,
            predictions
        )

        actual_change = np.diff(
            actual_retail
        )

        predicted_change = np.diff(
            predictions
        )

        change_mae = mean_absolute_error(
            actual_change,
            predicted_change
        )

        direction_accuracy = (
            np.sign(actual_change) ==
            np.sign(predicted_change)
        ).mean()

    else:

        r2 = np.nan
        change_mae = np.nan
        direction_accuracy = np.nan

    ##########################################################################
    # SAVE METRICS
    ##########################################################################

    window_metrics.append({

        "Window": window,

        "RMSE": rmse,

        "MAE": mae,

        "MAPE": mape,

        "R2": r2,

        "Change_MAE": change_mae,

        "Direction_Accuracy": direction_accuracy,

        "Time": toc - tic

    })

    ##########################################################################
    # SAVE PREDICTIONS
    ##########################################################################

    for d, actual, pred in zip(
            dates,
            actual_retail,
            predictions):

        prediction_table.append({

            "Window": window,

            "Date": d,

            "Actual_Retail": actual,

            "Predicted_Retail": pred

        })

    window += 1

##############################################################################
# RESULTS
##############################################################################

metrics = pd.DataFrame(
    window_metrics
)

predictions_df = pd.DataFrame(
    prediction_table
)

##############################################################################
# PRINT RESULTS
##############################################################################

print("\n")
print("=" * 60)

print(
    f"Model            : ARIMA{ARIMA_ORDER}"
)

print(
    f"Target           : Total Retail Price"
)

print(
    f"Forecast Horizon : "
    f"{FORECAST_HORIZON} month(s)"
)

print(
    f"Rolling Windows  : "
    f"{len(metrics)}"
)

print("=" * 60)

print(
    f"Average RMSE : "
    f"{metrics['RMSE'].mean():.2f}"
)

print(
    f"Average MAE  : "
    f"{metrics['MAE'].mean():.2f}"
)

print(
    f"Average MAPE : "
    f"{metrics['MAPE'].mean() * 100:.2f}%"
)

print(
    f"Average R²   : "
    f"{metrics['R2'].mean():.3f}"
)

print(
    f"Average Change MAE : "
    f"{metrics['Change_MAE'].mean():.2f}"
)

print(
    f"Average Direction Accuracy : "
    f"{metrics['Direction_Accuracy'].mean() * 100:.2f}%"
)

print(
    f"Average Training Time : "
    f"{metrics['Time'].mean():.4f} sec"
)

##############################################################################
# WINDOW RESULTS
##############################################################################

print("\nWindow Results\n")

print(metrics.to_string(index=False))

##############################################################################
# OPTIONAL
##############################################################################

# metrics.to_csv(
#     "arima_retail_rolling_metrics.csv",
#     index=False
# )

# predictions_df.to_csv(
#     "arima_retail_rolling_predictions.csv",
#     index=False
# )

##############################################################################
# PLOT FORECAST VS ACTUAL
##############################################################################

import matplotlib.pyplot as plt

predictions_df = (
    predictions_df
    .sort_values("Date")
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    predictions_df["Date"],
    predictions_df["Actual_Retail"],
    label="Actual",
    linewidth=2
)

plt.plot(
    predictions_df["Date"],
    predictions_df["Predicted_Retail"],
    label="Forecast",
    linewidth=2
)

plt.title(
    f"ARIMA{ARIMA_ORDER} "
    f"{FORECAST_HORIZON}-Month "
    f"Total Retail Price Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Retail Price (c/L)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()