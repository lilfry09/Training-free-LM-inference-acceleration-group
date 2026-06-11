"""Plot one or more metrics CSV files from results/ into figures/."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PLOTS = [
    ("ppl", "Perplexity", "ppl.png", None),
    ("tokens_per_second", "Tokens / second", "tokens_per_second.png", None),
    ("kv_cache_mb_after_prefill", "KV cache after prefill (MB)", "kv_cache_mb_after_prefill.png", None),
    ("kv_cache_compression_ratio_actual", "KV cache compression (%)", "kv_cache_compression_ratio.png", 100.0),
    ("process_rss_delta_mb", "Process RSS delta (MB)", "process_rss_delta_mb.png", None),
]

TRADEOFF_PLOTS = [
    ("ppl", "Perplexity", "compression_vs_ppl.png"),
    ("kv_cache_mb_after_prefill", "KV cache after prefill (MB)", "compression_vs_memory.png"),
]

METHOD_COMPARISON_PLOTS = [
    ("ppl", "wikitext", "Method vs WikiText PPL", "method_vs_wikitext_ppl.png"),
    ("ppl", "pg19", "Method vs PG19 PPL", "method_vs_pg19_ppl.png"),
    ("kv_cache_mb_after_prefill", None, "Method vs KV Cache Memory", "method_vs_kv_cache_memory.png"),
    ("tokens_per_second", None, "Method vs Tokens/s", "method_vs_tokens_per_second.png"),
]


def infer_experiment_label(path: Path, df: pd.DataFrame) -> str:
    values = df.loc[df["method"] == "snapkv", "snapkv_compression_ratio"].dropna().unique()
    if len(values) == 1:
        return f"cr={float(values[0]):.2f}"
    return path.stem


def load_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/evaluate.py before plotting.")
        frame = pd.read_csv(path)
        frame["experiment"] = infer_experiment_label(path, frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def annotate_bars(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str, scale: float | None) -> None:
    if metric not in df.columns:
        print(f"Skipping missing metric: {metric}")
        return

    plot_df = df.copy()
    y_col = metric
    if scale is not None:
        y_col = f"{metric}_scaled"
        plot_df[y_col] = plot_df[metric] * scale

    if "experiment" in plot_df.columns and plot_df["experiment"].nunique() > 1:
        plot_df["series"] = plot_df["method"] + " (" + plot_df["experiment"] + ")"
        hue_col = "series"
        legend_title = "Method / experiment"
        width = 9
    else:
        hue_col = "method"
        legend_title = "Method"
        width = 7

    plt.figure(figsize=(width, 4.4))
    ax = sns.barplot(data=plot_df, x="dataset", y=y_col, hue=hue_col)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    annotate_bars(ax)
    ax.legend(title=legend_title)
    plt.tight_layout()
    out_path = FIGURES / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_tradeoff(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    required = {"method", "dataset", "kv_cache_compression_ratio_actual", metric}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping {filename}; missing columns: {sorted(missing)}")
        return

    plot_df = df[df["method"] == "snapkv"].copy()
    if plot_df.empty:
        print(f"Skipping {filename}; no snapkv rows found.")
        return

    plot_df["compression_percent"] = plot_df["kv_cache_compression_ratio_actual"] * 100.0
    plot_df = plot_df.sort_values(["dataset", "compression_percent"])
    x_col = "compression_percent"
    style_col = None
    markers = True
    if metric == "kv_cache_mb_after_prefill":
        datasets = sorted(plot_df["dataset"].unique())
        offsets = {
            dataset: (idx - (len(datasets) - 1) / 2) * 0.6
            for idx, dataset in enumerate(datasets)
        }
        plot_df["compression_percent_offset"] = plot_df["compression_percent"] + plot_df["dataset"].map(offsets)
        x_col = "compression_percent_offset"
        style_col = "dataset"
        markers = {"pg19": "s", "wikitext": "o"}

    plt.figure(figsize=(7, 4.4))
    ax = sns.lineplot(
        data=plot_df,
        x=x_col,
        y=metric,
        hue="dataset",
        style=style_col,
        markers=markers,
        dashes=False,
        errorbar=None,
    )
    ax.set_xlabel("KV cache compression (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Compression vs {ylabel}")
    ax.legend(title="Dataset")
    plt.tight_layout()
    out_path = FIGURES / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_method_comparison(
    df: pd.DataFrame,
    metric: str,
    dataset_name: str | None,
    ylabel: str,
    filename: str,
    scale: float | None = None,
) -> None:
    if dataset_name is None:
        plot_df = df.copy()
    else:
        plot_df = df[df["dataset"] == dataset_name]
        if plot_df.empty:
            print(f"Skipping {filename}; no rows for dataset={dataset_name}.")
            return

    if metric not in plot_df.columns:
        print(f"Skipping {filename}; missing metric: {metric}")
        return

    y_col = metric
    if scale is not None:
        y_col = f"{metric}_scaled"
        plot_df = plot_df.copy()
        plot_df[y_col] = plot_df[metric] * scale

    plt.figure(figsize=(6.6, 4.0))
    hue_col = "method"
    if dataset_name is None:
        hue_col = "dataset"
    ax = sns.barplot(data=plot_df, x="method", y=y_col, hue=hue_col)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    annotate_bars(ax)
    if dataset_name is None and "dataset" in plot_df.columns:
        ax.legend(title="Dataset")
    else:
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    out_path = FIGURES / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SnapKV evaluation metrics.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[RESULTS / "metrics.csv"],
        help="Input metrics CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES.mkdir(exist_ok=True)
    df = load_metrics(args.inputs)
    required = {"dataset", "method"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    sns.set_theme(style="whitegrid", context="paper")
    for metric, ylabel, filename, scale in PLOTS:
        plot_metric(df, metric, ylabel, filename, scale)
    for metric, dataset_name, ylabel, filename in METHOD_COMPARISON_PLOTS:
        plot_method_comparison(df, metric, dataset_name, ylabel, filename)
    for metric, ylabel, filename in TRADEOFF_PLOTS:
        plot_tradeoff(df, metric, ylabel, filename)

    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
