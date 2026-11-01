import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

from timestral import TimestralForecaster


def load_series(data_path, series_col, time_col):
    series = defaultdict(list)
    with Path(data_path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            series[row[series_col]].append(row)
    for rows in series.values():
        rows.sort(key=lambda row: float(row[time_col]))
    return series


def load_forecaster(model_name, device):
    return TimestralForecaster.from_pretrained(
        model_name,
        device=device,
        cache_dir=str(ROOT / ".cache" / "huggingface"),
    )


def predict_timestral(
    series,
    forecaster,
    context_hours,
    forecast_hours,
    model_name,
    series_col,
    time_col,
    target_col,
    metadata_cols,
    past_covariates,
    known_covariates,
):
    step_seconds = float(np.median([
        float(b[time_col]) - float(a[time_col])
        for rows in series.values()
        for a, b in zip(rows, rows[1:])
    ]))
    forecast_steps = int(round(forecast_hours * 3600 / step_seconds))
    origin_seconds = context_hours * 3600

    rows_out = []
    for series_id in sorted(series):
        rows = series[series_id]
        values = np.array([float(row[target_col]) for row in rows])
        all_times = np.array([float(row[time_col]) for row in rows])
        origin_index = int(np.searchsorted(all_times, origin_seconds, side="right"))
        context = values[:origin_index][None, None, :]
        measured = values[origin_index:origin_index + forecast_steps]
        times = all_times[origin_index:origin_index + forecast_steps]

        past = np.array([[
            [float(row[col]) for row in rows[:origin_index]]
            for col in past_covariates
        ]]) if past_covariates else None
        known = np.array([[
            [float(row[col]) for row in rows[:origin_index + forecast_steps]]
            for col in known_covariates
        ]]) if known_covariates else None

        forecast = forecaster.forecast(
            context,
            prediction_length=forecast_steps,
            past_covariates=past,
            known_covariates=known,
        )
        predicted = forecast.median[0, 0].cpu().numpy()
        predicted_q05 = forecast.quantile(0.05)[0, 0].cpu().numpy()
        predicted_q95 = forecast.quantile(0.95)[0, 0].cpu().numpy()

        for j in range(forecast_steps):
            row = rows[origin_index + j]
            row_out = {series_col: series_id}
            row_out.update({name: row[name] for name in metadata_cols})
            row_out["context_hours"] = context_hours
            row_out["forecast_hours"] = forecast_hours
            row_out["model_name"] = model_name
            row_out["past_covariates_used"] = ",".join(past_covariates)
            row_out["known_covariates_used"] = ",".join(known_covariates)
            row_out["elapsed_h"] = float(times[j]) / 3600
            row_out["measured"] = float(measured[j])
            row_out["predicted"] = float(predicted[j])
            row_out["predicted_q05"] = float(predicted_q05[j])
            row_out["predicted_q95"] = float(predicted_q95[j])
            rows_out.append(row_out)
    return rows_out


def compute_metrics(
    predictions,
    batch_col="series_id",
    measured_col="measured",
    predicted_col="predicted",
):
    groups = defaultdict(list)
    for row in predictions:
        groups[row[batch_col]].append(row)

    fieldnames = list(predictions[0].keys())
    excluded_cols = {"elapsed_h", measured_col, predicted_col}
    id_cols = [batch_col] + [
        col for col in fieldnames
        if col != batch_col
        and col not in excluded_cols
        and all(len({row[col] for row in rows}) == 1 for rows in groups.values())
    ]

    metrics_rows = []
    p = 1
    for batch, rows in sorted(groups.items()):
        measured = np.array([float(row[measured_col]) for row in rows])
        predicted = np.array([float(row[predicted_col]) for row in rows])
        n = measured.size
        sse = np.sum((predicted - measured) ** 2)
        sst = np.sum((measured - measured.mean()) ** 2)
        r2 = 1 - sse / sst
        metrics_row = {col: rows[0][col] for col in id_cols}
        metrics_row.update({
            "n": n,
            "RMSE": np.sqrt(sse / n),
            "BIAS": np.mean(predicted - measured),
            "R2": r2,
            "Adjusted_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        })
        metrics_rows.append(metrics_row)
    return metrics_rows


def save_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

