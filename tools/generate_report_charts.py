#!/usr/bin/env python3
"""Generate report charts from cached LA-M2MRF experiment metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work_dirs" / "la_m2mrf" / "report_figures"
METRICS_CACHE_DIR = REPO_ROOT / "work_dirs" / "la_m2mrf" / "notebook_artifacts" / "metrics_cache"
RESULTS_CSV = REPO_ROOT / "work_dirs" / "la_m2mrf" / "results" / "idrid_comparison_table.csv"
FONT_PATH = REPO_ROOT / "NotoSansTC-Regular.otf"
CJK_FONT_PATH_HINTS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/mnt/c/Windows/Fonts/msjh.ttc"),
    Path("/mnt/c/Windows/Fonts/msjhbd.ttc"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/mingliu.ttc"),
]

FIG1_SOURCES = [
    {
        "label": "Official\nCheckpoint Eval",
        "cache_key": "official_released_checkpoint",
        "csv_run_name": "official_checkpoint_eval",
    },
    {
        "label": "Finetune\nBaseline",
        "cache_key": "finetune_baseline",
    },
    {
        "label": "Scratch\nBaseline",
        "cache_key": "baseline_scratch_bs3_fp16",
    },
]

FIG2_SOURCES = [
    ("M2MRF-C / Control", "finetune_baseline"),
    ("LA-M2MRF-S", "finetune_la_sampler"),
    ("LA-M2MRF-W", "finetune_weighted_dice"),
    ("LA-M2MRF-SW", "finetune_la_sampler_weighted_dice"),
]

LESION_CLASSES = ["EX", "HE", "SE", "MA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quantitative report charts from cached experiment metrics."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PNG charts will be written.",
    )
    parser.add_argument(
        "--include-fig3",
        action="store_true",
        help="Also generate Figure 3 via tools/generate_report_fig3.py.",
    )
    parser.add_argument(
        "--fig3-cases",
        nargs="+",
        default=None,
        help="Optional case ids passed through to Figure 3 generation.",
    )
    return parser.parse_args()


def find_font_path() -> Path | None:
    if FONT_PATH.is_file():
        return FONT_PATH

    for font_path in CJK_FONT_PATH_HINTS:
        if font_path.is_file():
            return font_path

    for font_file in fm.fontManager.findSystemFonts():
        lower_name = Path(font_file).name.lower()
        if any(keyword in lower_name for keyword in ["notosanscjk", "msjh", "msyh", "mingliu", "simsun"]):
            return Path(font_file)
    return None


def configure_fonts() -> None:
    font_path = find_font_path()
    if font_path is not None:
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft JhengHei",
            "PingFang HK",
            "Taipei Sans TC Beta",
            "Noto Sans CJK TC",
            "SimHei",
            "Microsoft YaHei",
            "sans-serif",
        ]
    plt.rcParams["axes.unicode_minus"] = False


def load_results_csv() -> Dict[str, dict]:
    if not RESULTS_CSV.is_file():
        return {}

    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["run_name"]: row for row in reader if row.get("run_name")}


def load_metrics_cache(cache_key: str) -> dict:
    path = METRICS_CACHE_DIR / f"{cache_key}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics cache not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_percent(value) -> float:
    if value is None:
        raise ValueError("Metric value is missing.")

    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100.0
    return numeric


def get_summary_metric(cache_key: str, metric_name: str, csv_run_name: str | None = None) -> float:
    cache = load_metrics_cache(cache_key)
    value = cache.get(metric_name)
    if value is not None:
        return to_percent(value)

    if csv_run_name:
        csv_rows = load_results_csv()
        row = csv_rows.get(csv_run_name)
        if row and row.get(metric_name):
            return to_percent(row[metric_name])

    raise KeyError(
        f"Could not find summary metric '{metric_name}' for cache '{cache_key}'."
    )


def get_per_class_iou(cache_key: str, class_name: str) -> float:
    cache = load_metrics_cache(cache_key)
    per_class = cache.get("per_class", {})
    class_metrics = per_class.get(class_name)
    if not class_metrics or class_metrics.get("IoU") is None:
        raise KeyError(
            f"Could not find per-class IoU for class '{class_name}' in cache '{cache_key}'."
        )
    return to_percent(class_metrics["IoU"])


def collect_fig1_data() -> dict:
    labels = []
    miou = []
    maupr = []

    for source in FIG1_SOURCES:
        labels.append(source["label"])
        miou.append(
            get_summary_metric(
                cache_key=source["cache_key"],
                metric_name="mIoU",
                csv_run_name=source.get("csv_run_name"),
            )
        )
        maupr.append(
            get_summary_metric(
                cache_key=source["cache_key"],
                metric_name="mAUPR",
                csv_run_name=source.get("csv_run_name"),
            )
        )

    return {"labels": labels, "mIoU": miou, "mAUPR": maupr}


def collect_fig2_data() -> dict:
    series = {}
    for legend, cache_key in FIG2_SOURCES:
        series[legend] = [get_per_class_iou(cache_key, class_name) for class_name in LESION_CLASSES]
    return series


def metric_ylim(values: Iterable[float], padding: float = 0.6) -> tuple[float, float]:
    values = list(values)
    lower = min(values) - padding
    upper = max(values) + padding
    return lower, upper


def draw_figure1(output_dir: Path) -> Path:
    data = collect_fig1_data()
    labels = data["labels"]
    miou = data["mIoU"]
    maupr = data["mAUPR"]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rects1 = ax1.bar(x, miou, width, color="#4A90E2")
    ax1.set_ylabel("mIoU (%)")
    ax1.set_title("mIoU Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(*metric_ylim(miou))
    ax1.bar_label(rects1, padding=3, fmt="%.2f")

    rects2 = ax2.bar(x, maupr, width, color="#F5A623")
    ax2.set_ylabel("mAUPR (%)")
    ax2.set_title("mAUPR Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(*metric_ylim(maupr))
    ax2.bar_label(rects2, padding=3, fmt="%.2f")

    fig.suptitle("M2MRF Baseline Variants Quantitative Comparison", fontsize=14)
    fig.tight_layout()

    output_path = output_dir / "圖1_M2MRF-C官方結果與本機復現結果比較.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def draw_figure2(output_dir: Path) -> Path:
    data = collect_fig2_data()
    labels = LESION_CLASSES
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#D9D9D9", "#7FB3D5", "#82E0AA", "#F1948A"]
    offsets = [-1.5, -0.5, 0.5, 1.5]

    all_values: List[float] = []
    for (legend, values), color, offset in zip(data.items(), colors, offsets):
        bars = ax.bar(x + offset * width, values, width, label=legend, color=color)
        ax.bar_label(bars, padding=3, fontsize=9, fmt="%.2f")
        all_values.extend(values)

    ax.set_ylabel("IoU (%)")
    ax.set_title("Per-lesion IoU Comparison Across Methods", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    lower, upper = metric_ylim(all_values, padding=2.0)
    ax.set_ylim(max(0.0, lower), min(100.0, upper))
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path = output_dir / "圖2_不同改良方法於四類病灶之IoU表現比較.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    configure_fonts()

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    print("Generating quantitative charts from cached metrics.")
    print(f"Output directory: {output_dir}")
    print("Note: this script does not read GT masks.")
    print("For GT / prediction overlays, use tools/generate_report_fig3.py instead.")

    fig1_path = draw_figure1(output_dir)
    fig2_path = draw_figure2(output_dir)

    print(f"Saved Figure 1 to: {fig1_path}")
    print(f"Saved Figure 2 to: {fig2_path}")

    if args.include_fig3:
        from generate_report_fig3 import create_fig3

        fig3_path = create_fig3(output_dir=output_dir, case_ids=args.fig3_cases)
        print(f"Saved Figure 3 to: {fig3_path}")


if __name__ == "__main__":
    main()
