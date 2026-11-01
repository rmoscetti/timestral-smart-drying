import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROLLING_PREDICTIONS = ROOT / "results" / "01_timestral_rolling_predictions.csv"
INPUT_ROLLING_METRICS = ROOT / "results" / "01_timestral_rolling_metrics.csv"
INPUT_VALIDATION_ALL = ROOT / "results" / "02_timestral_validation_all.csv"
OUTPUT_VALIDATION_SUMMARY = ROOT / "results" / "03_timestral_validation_summary.csv"
OUTPUT_DIR = ROOT / "results" / "plots"
DPI = 300
plot_context_hours = 3.0


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Nessuna riga in {path}")
    return rows


def ensure_columns(rows, columns, path):
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Colonne mancanti in {path}: {', '.join(missing)}")


def to_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_values(rows, key, absolute=False):
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if absolute:
        values = [abs(value) for value in values]
    return np.asarray(values, dtype=float)


def paired_values(rows, x_key, y_key):
    xs = []
    ys = []
    for row in rows:
        x = to_float(row.get(x_key))
        y = to_float(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def row_values(rows, key):
    values = [to_float(row.get(key)) for row in rows]
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def save_figure(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(predictions):
    contexts = sorted(
        {
            value
            for value in (to_float(row.get("context_hours")) for row in predictions)
            if value is not None
        }
    )
    if not contexts:
        return

    context = min(contexts, key=lambda value: (abs(value - plot_context_hours), value))
    rows = [
        row
        for row in predictions
        if to_float(row.get("context_hours")) == context
    ]
    rows.sort(key=lambda row: float(row["elapsed_h"]))

    time = row_values(rows, "elapsed_h")
    measured = row_values(rows, "measured")
    predicted = row_values(rows, "predicted")
    q05 = row_values(rows, "predicted_q05")
    q95 = row_values(rows, "predicted_q95")

    series_id = rows[0].get("series_id", "")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if np.isfinite(q05).any() and np.isfinite(q95).any():
        ax.fill_between(
            time,
            q05,
            q95,
            color="tab:blue",
            alpha=0.18,
            linewidth=0,
            label="q05-q95",
        )
    ax.plot(
        time,
        measured,
        color="black",
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="Measured",
    )
    ax.plot(time, predicted, color="tab:blue", linewidth=1.2, label="Predicted")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("MCdb (g H2O/g DM)")
    ax.set_title(f"{series_id}: rolling forecast at context = {context:g} h")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "03_rolling_timeseries_measured_predicted_q05_q95.png")


def plot_target_convergence(metrics):
    rows = sorted(metrics, key=lambda row: float(row["context_hours"]))
    context = row_values(rows, "context_hours")
    predicted = row_values(rows, "predicted_target_h")
    observed = row_values(rows, "observed_target_h")
    q05 = row_values(rows, "predicted_target_h_q05")
    q95 = row_values(rows, "predicted_target_h_q95")

    series_id = rows[0].get("series_id", "")
    target = to_float(rows[0].get("target_mcdb"))
    target_label = "n/a" if target is None else f"{target:g}"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if np.isfinite(q05).any() and np.isfinite(q95).any():
        ax.fill_between(
            context,
            q05,
            q95,
            color="tab:blue",
            alpha=0.18,
            linewidth=0,
            label="q05-q95",
        )
    ax.plot(
        context,
        predicted,
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Predicted median",
    )
    ax.plot(
        context,
        observed,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Observed target",
    )
    ax.set_xlabel("Context duration (h)")
    ax.set_ylabel("Time to target (h)")
    ax.set_title(f"{series_id}: target convergence (MCdb = {target_label})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "03_rolling_target_convergence.png")


def plot_rolling_metric(metrics, y_key, ylabel, filename, zero_line=False):
    context, y = paired_values(metrics, "context_hours", y_key)
    if context.size == 0:
        return
    order = np.argsort(context)
    context = context[order]
    y = y[order]

    series_id = metrics[0].get("series_id", "")
    target = to_float(metrics[0].get("target_mcdb"))
    target_label = "n/a" if target is None else f"{target:g}"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(context, y, color="tab:blue", marker="o", markersize=3, linewidth=1.2)
    if zero_line:
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Context duration (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{series_id}: {ylabel} (MCdb = {target_label})")
    ax.grid(alpha=0.25)
    save_figure(fig, filename)


def mean_value(values):
    return float(np.mean(values)) if values.size else None


def median_value(values):
    return float(np.median(values)) if values.size else None


def quantile_value(values, quantile):
    return float(np.percentile(values, quantile)) if values.size else None


def summarize_validation(rows):
    grouped = defaultdict(list)
    for row in rows:
        target = to_float(row.get("target_mcdb"))
        context = to_float(row.get("context_hours"))
        if target is None or context is None:
            continue
        grouped[(target, context)].append(row)

    summary = []
    for (target, context), group in sorted(
        grouped.items(),
        key=lambda item: (-item[0][0], item[0][1]),
    ):
        target_abs = finite_values(group, "target_error_min", absolute=True)
        mcdb_abs = finite_values(group, "mcdb_error_at_predicted_stop", absolute=True)
        rmse = finite_values(group, "RMSE_to_target")
        bias = finite_values(group, "BIAS_to_target")
        width = finite_values(group, "target_interval_width_min")
        covered = finite_values(group, "target_covered")

        summary.append(
            {
                "target_mcdb": target,
                "context_hours": context,
                "n": len(group),
                "mean_abs_target_error_min": mean_value(target_abs),
                "median_abs_target_error_min": median_value(target_abs),
                "q25_abs_target_error_min": quantile_value(target_abs, 25),
                "q75_abs_target_error_min": quantile_value(target_abs, 75),
                "mean_abs_mcdb_error": mean_value(mcdb_abs),
                "median_abs_mcdb_error": median_value(mcdb_abs),
                "q25_abs_mcdb_error": quantile_value(mcdb_abs, 25),
                "q75_abs_mcdb_error": quantile_value(mcdb_abs, 75),
                "mean_RMSE_to_target": mean_value(rmse),
                "mean_BIAS_to_target": mean_value(bias),
                "mean_target_interval_width_min": mean_value(width),
                "target_coverage": mean_value(covered),
            }
        )
    return summary


def save_validation_summary(summary):
    columns = [
        "target_mcdb",
        "context_hours",
        "n",
        "mean_abs_target_error_min",
        "median_abs_target_error_min",
        "q25_abs_target_error_min",
        "q75_abs_target_error_min",
        "mean_abs_mcdb_error",
        "median_abs_mcdb_error",
        "q25_abs_mcdb_error",
        "q75_abs_mcdb_error",
        "mean_RMSE_to_target",
        "mean_BIAS_to_target",
        "mean_target_interval_width_min",
        "target_coverage",
    ]
    with OUTPUT_VALIDATION_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)


def plot_target_summary(
    summary,
    y_key,
    ylabel,
    suffix,
    band_keys=None,
    y_limits=None,
    title=None,
):
    grouped = defaultdict(list)
    for row in summary:
        target = to_float(row.get("target_mcdb"))
        if target is not None:
            grouped[target].append(row)

    for target in sorted(grouped, reverse=True):
        rows = sorted(grouped[target], key=lambda row: float(row["context_hours"]))
        context = row_values(rows, "context_hours")
        y = row_values(rows, y_key)
        target_label = f"{target:g}"

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        if band_keys is not None:
            low = row_values(rows, band_keys[0])
            high = row_values(rows, band_keys[1])
            if np.isfinite(low).any() and np.isfinite(high).any():
                ax.fill_between(
                    context,
                    low,
                    high,
                    color="tab:blue",
                    alpha=0.18,
                    linewidth=0,
                    label="Q25-Q75",
                )
        ax.plot(
            context,
            y,
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="Median" if band_keys is not None else "Mean",
        )
        ax.set_xlabel("Context duration (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Target MCdb = {target_label}: {title or ylabel}")
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        save_figure(fig, f"03_target_{target_label}_{suffix}.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    predictions = load_rows(INPUT_ROLLING_PREDICTIONS)
    ensure_columns(
        predictions,
        [
            "series_id",
            "context_hours",
            "elapsed_h",
            "measured",
            "predicted",
            "predicted_q05",
            "predicted_q95",
        ],
        INPUT_ROLLING_PREDICTIONS,
    )

    metrics = load_rows(INPUT_ROLLING_METRICS)
    ensure_columns(
        metrics,
        [
            "series_id",
            "context_hours",
            "target_mcdb",
            "predicted_target_h",
            "observed_target_h",
            "predicted_target_h_q05",
            "predicted_target_h_q95",
            "target_error_min",
            "mcdb_error_at_predicted_stop",
            "target_interval_width_min",
        ],
        INPUT_ROLLING_METRICS,
    )

    validation = load_rows(INPUT_VALIDATION_ALL)
    ensure_columns(
        validation,
        [
            "context_hours",
            "target_mcdb",
            "target_error_min",
            "mcdb_error_at_predicted_stop",
            "target_interval_width_min",
            "target_covered",
            "RMSE_to_target",
            "BIAS_to_target",
        ],
        INPUT_VALIDATION_ALL,
    )

    plot_timeseries(predictions)
    plot_target_convergence(metrics)
    plot_rolling_metric(
        metrics,
        "target_error_min",
        "Target time error (min)",
        "03_rolling_target_time_error.png",
        zero_line=True,
    )
    plot_rolling_metric(
        metrics,
        "mcdb_error_at_predicted_stop",
        "MCdb error at predicted stop (g H2O/g DM)",
        "03_rolling_mcdb_error_at_predicted_stop.png",
        zero_line=True,
    )
    plot_rolling_metric(
        metrics,
        "target_interval_width_min",
        "Target interval width (min)",
        "03_rolling_target_interval_width.png",
    )

    summary = summarize_validation(validation)
    save_validation_summary(summary)

    plot_target_summary(
        summary,
        "median_abs_target_error_min",
        "Median absolute target time error (min)",
        "target_time_error",
        band_keys=("q25_abs_target_error_min", "q75_abs_target_error_min"),
        title="absolute target time error",
    )
    plot_target_summary(
        summary,
        "median_abs_mcdb_error",
        "Median absolute MCdb error (g H2O/g DM)",
        "mcdb_error",
        band_keys=("q25_abs_mcdb_error", "q75_abs_mcdb_error"),
        title="absolute MCdb error at predicted stop",
    )
    plot_target_summary(
        summary,
        "mean_RMSE_to_target",
        "Mean RMSE to target (g H2O/g DM)",
        "rmse_to_target",
        title="RMSE to target",
    )
    plot_target_summary(
        summary,
        "mean_target_interval_width_min",
        "Mean target interval width (min)",
        "target_interval_width",
        title="target interval width",
    )
    plot_target_summary(
        summary,
        "target_coverage",
        "Observed target within q05-q95 interval (proportion)",
        "coverage",
        y_limits=(0.0, 1.0),
        title="Observed target within q05-q95 interval",
    )


if __name__ == "__main__":
    main()
