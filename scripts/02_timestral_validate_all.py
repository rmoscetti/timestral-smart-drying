from pathlib import Path

from timestral_utils import (
    compute_metrics,
    load_forecaster,
    load_series,
    predict_timestral,
    save_csv,
)

ROOT = Path(__file__).resolve().parents[1]


def first_crossing_h(points, target_mcdb):
    for (time_a, value_a), (time_b, value_b) in zip(points, points[1:]):
        if value_a >= target_mcdb and value_b <= target_mcdb and value_a != value_b:
            return time_a + (value_a - target_mcdb) / (value_a - value_b) * (time_b - time_a)
    return None


def interpolate_h(points, time_h):
    for (time_a, value_a), (time_b, value_b) in zip(points, points[1:]):
        if time_a <= time_h <= time_b:
            return value_a + (value_b - value_a) * (time_h - time_a) / (time_b - time_a)
    return None


model_name = "mistralai/Timestral-209M"
device = "cuda"
series_col = "series_id"
time_col = "elapsed_s"
target_col = "MCdb_gH2O_gDM"
metadata_cols = ["cultivar", "replicate", "pef", "blanching"]
past_covariates = []
known_covariates = []
data_path = ROOT / "data" / "timestral_potato_drying_MCdb.csv"
output_path = ROOT / "results" / "02_timestral_validation_all.csv"
report_columns = [
    "series_id",
    "cultivar",
    "replicate",
    "pef",
    "blanching",
    "model_name",
    "past_covariates_used",
    "known_covariates_used",
    "context_hours",
    "context_step_hours",
    "target_mcdb",
    "current_time_h",
    "predicted_target_h",
    "observed_target_h",
    "target_error_min",
    "observed_mcdb_at_predicted_stop",
    "mcdb_error_at_predicted_stop",
    "predicted_target_h_q05",
    "predicted_target_h_q95",
    "target_interval_width_min",
    "target_covered",
    "RMSE_to_target",
    "BIAS_to_target",
]
forecast_hours = 1
forecast_strategy = "remaining"
final_hour = 10
context_step_hours = 0.25
target_mcdb_values = [2.5, 2.0, 1.6, 1.2, 1.0]

series = load_series(data_path, series_col, time_col)
forecaster = load_forecaster(model_name, device)

rows_out = []
for series_id in sorted(series):
    rows = series[series_id]
    selected_series = {series_id: rows}
    last_time = float(rows[-1][time_col])
    available_hours = int(last_time // 3600)
    max_context_hours = available_hours - forecast_hours if forecast_strategy == "fixed" else final_hour - 1
    n_context_steps = int(max_context_hours / context_step_hours)

    for context_step in range(1, n_context_steps + 1):
        context_hours = context_step * context_step_hours
        origin_row = max(
            (row for row in rows if float(row[time_col]) <= context_hours * 3600),
            key=lambda row: float(row[time_col]),
        )
        origin_mcdb = float(origin_row[target_col])
        active_targets = [target_mcdb for target_mcdb in target_mcdb_values if origin_mcdb > target_mcdb]
        if not active_targets:
            break

        current_time_h = float(origin_row[time_col]) / 3600
        current_forecast_hours = forecast_hours if forecast_strategy == "fixed" else final_hour - context_hours
        predictions = predict_timestral(
            selected_series,
            forecaster,
            context_hours,
            current_forecast_hours,
            model_name,
            series_col,
            time_col,
            target_col,
            metadata_cols,
            past_covariates,
            known_covariates,
        )

        base_row = compute_metrics(predictions)[0]
        origin_point = (current_time_h, origin_mcdb)
        predicted_points = [origin_point] + [
            (float(row["elapsed_h"]), float(row["predicted"]))
            for row in predictions
        ]
        observed_points = [origin_point] + [
            (float(row["elapsed_h"]), float(row["measured"]))
            for row in predictions
        ]
        predicted_points_q05 = [origin_point] + [
            (float(row["elapsed_h"]), float(row["predicted_q05"]))
            for row in predictions
        ]
        predicted_points_q95 = [origin_point] + [
            (float(row["elapsed_h"]), float(row["predicted_q95"]))
            for row in predictions
        ]

        for target_mcdb in active_targets:
            predicted_target_h = first_crossing_h(predicted_points, target_mcdb)
            predicted_target_h_q05 = first_crossing_h(predicted_points_q05, target_mcdb)
            predicted_target_h_q95 = first_crossing_h(predicted_points_q95, target_mcdb)
            observed_target_h = first_crossing_h(observed_points, target_mcdb)
            observed_mcdb_at_predicted_stop = (
                interpolate_h(observed_points, predicted_target_h)
                if predicted_target_h is not None
                else None
            )
            mcdb_error_at_predicted_stop = (
                observed_mcdb_at_predicted_stop - target_mcdb
                if observed_mcdb_at_predicted_stop is not None
                else None
            )
            target_covered = None
            if observed_target_h is not None and predicted_target_h_q05 is not None and predicted_target_h_q95 is not None:
                target_covered = int(predicted_target_h_q05 <= observed_target_h <= predicted_target_h_q95)

            to_target_metrics = {}
            if observed_target_h is not None:
                to_target_predictions = [
                    row for row in predictions
                    if float(row["elapsed_h"]) <= observed_target_h
                ]
                if to_target_predictions:
                    to_target_metrics = compute_metrics(to_target_predictions)[0]

            row = dict(base_row)
            row.update({
                "target_mcdb": target_mcdb,
                "context_step_hours": context_step_hours,
                "current_time_h": current_time_h,
                "predicted_target_h": predicted_target_h,
                "predicted_target_h_q05": predicted_target_h_q05,
                "predicted_target_h_q95": predicted_target_h_q95,
                "observed_target_h": observed_target_h,
                "target_error_min": (predicted_target_h - observed_target_h) * 60 if predicted_target_h is not None and observed_target_h is not None else None,
                "target_interval_width_min": abs(predicted_target_h_q95 - predicted_target_h_q05) * 60 if predicted_target_h_q05 is not None and predicted_target_h_q95 is not None else None,
                "observed_mcdb_at_predicted_stop": observed_mcdb_at_predicted_stop,
                "mcdb_error_at_predicted_stop": mcdb_error_at_predicted_stop,
                "target_covered": target_covered,
                "RMSE_to_target": to_target_metrics.get("RMSE"),
                "BIAS_to_target": to_target_metrics.get("BIAS"),
            })
            rows_out.append(row)

rows_out.sort(key=lambda row: (row["series_id"], row["target_mcdb"], row["context_hours"]))
rows_out = [
    {column: row.get(column) for column in report_columns}
    for row in rows_out
]

save_csv(rows_out, output_path)
