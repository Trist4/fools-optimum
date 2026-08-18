from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from prophet import Prophet
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def run_prophet_model(df, feature_cols, target_col="Total_Fuel_Price", FORECAST_HORIZON = 1, TRAIN_RATIO = 0.8,
                       include_2026=True, plot=True):
    """
    Fits and evaluates a Prophet model using rolling-origin (walk-forward) validation.

    At each step, the model is trained on all data known up to that point, and
    predicts FORECAST_HORIZON months ahead. Once that actual outcome "arrives"
    (i.e. the loop advances past it), it's folded into the training set for the
    next refit — so the test set is walked forward one prediction at a time
    rather than forecast in one shot.

    Uses global TRAIN_RATIO / TEST_RATIO to define the initial train/test split
    point, and FORECAST_HORIZON for how many months ahead each prediction targets.

    Parameters
    ----------
    df           : master dataframe (e.g. master_diesel_df)
    feature_cols : list of selected regressor column names
    target_col   : target price column
    include_2026 : if False, drops all rows from 2026 onward before modelling
    plot         : if True, shows an actual vs predicted plot for the test set

    Returns
    -------
    dict with rmse, mae, r2, per-step predictions, and the final fitted model
    """
    feature_cols = [c for c in feature_cols if c not in ("Date", target_col)]

    data = df.copy()
    if not include_2026:
        data = data[data["Date"] < "2026-01-01"]

    prophet_df = data[["Date", target_col] + feature_cols].copy()
    prophet_df = prophet_df.rename(columns={"Date": "ds", target_col: "y"})
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # Only the target is shifted forward — features stay at row t
    prophet_df["y"] = prophet_df["y"].shift(-FORECAST_HORIZON)
    prophet_df = prophet_df.dropna().reset_index(drop=True)

    n_total = len(prophet_df)
    split_idx = int(round(n_total * TRAIN_RATIO))

    if split_idx >= n_total:
        raise ValueError("Test set is empty — check TRAIN_RATIO/TEST_RATIO and data length.")

    dates, y_true_list, y_pred_list, y_lower_list, y_upper_list = [], [], [], [], []

    # Walk forward one test point at a time, expanding the training window each step
    for i in range(split_idx, n_total):
        train_fold = prophet_df.iloc[:i]        # everything known up to (not including) row i
        test_row = prophet_df.iloc[[i]]          # single point being predicted this step

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for col in feature_cols:
            model.add_regressor(col)
        model.fit(train_fold)

        fc = model.predict(test_row[["ds"] + feature_cols])

        dates.append(test_row["ds"].values[0])
        y_true_list.append(test_row["y"].values[0])
        y_pred_list.append(fc["yhat"].values[0])
        y_lower_list.append(fc["yhat_lower"].values[0])
        y_upper_list.append(fc["yhat_upper"].values[0])


    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    results_df = pd.DataFrame({
        "ds": dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "yhat_lower": y_lower_list,
        "yhat_upper": y_upper_list,
    })

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"include_2026={include_2026} | horizon={FORECAST_HORIZON}m | "
          f"walk-forward steps={len(results_df)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(results_df["ds"], results_df["y_true"], label="Actual", marker="o")
        ax.plot(results_df["ds"], results_df["y_pred"], label="Predicted", marker="o")
        ax.fill_between(results_df["ds"], results_df["yhat_lower"], results_df["yhat_upper"], alpha=0.2)
        ax.legend()
        ax.set_title(f"Prophet — {target_col} rolling {FORECAST_HORIZON}m-ahead (include_2026={include_2026})")
        plt.tight_layout()
        plt.show()

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model": model,          # final model, fit on all data up to the last test point
        "results_df": results_df,
        "train_df": train_fold
    }