#!/usr/bin/env python3
"""Generate the qualitative comparison chart for the LA-M2MRF report."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "OpenCV is required for tools/generate_report_fig3.py. "
        "Please run this script in the la-m2mrf environment."
    ) from exc

try:
    import torch
    from mmseg.apis import inference_segmentor, init_segmentor
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "mmcv / torch / mmseg are required for tools/generate_report_fig3.py. "
        "Please run this script in the la-m2mrf environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT / "../data/IDRID").resolve()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work_dirs" / "la_m2mrf" / "report_figures"
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
PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
    ],
    dtype=np.uint8,
)
DISPLAY_COLORS = {
    1: np.array([255, 255, 0], dtype=np.uint8),  # EX
    2: np.array([255, 0, 0], dtype=np.uint8),    # HE
    3: np.array([0, 0, 255], dtype=np.uint8),    # SE
    4: np.array([0, 255, 0], dtype=np.uint8),    # MA
}
DEFAULT_CASES = ["IDRiD_60", "IDRiD_68"]
DISPLAY_HEIGHT = 320

FIG3_TEXT = {
    "zh": {
        "figure_title": "不同方法於眼底病灶分割結果之視覺化比較",
        "legend_ex": "EX (硬性滲出物)",
        "legend_he": "HE (出血)",
        "legend_se": "SE (軟性滲出物)",
        "legend_ma": "MA (微血管瘤)",
        "output_name": "圖3_不同方法視覺化比較_排版.png",
    },
    "en": {
        "figure_title": "Qualitative Comparison of Retinal Lesion Segmentation Results Across Methods",
        "legend_ex": "EX (Hard Exudates)",
        "legend_he": "HE (Hemorrhages)",
        "legend_se": "SE (Soft Exudates)",
        "legend_ma": "MA (Microaneurysms)",
        "output_name": "Figure_3_Qualitative_Comparison.png",
    },
}

EXPERIMENTS = [
    {
        "title": "M2MRF-C / Control",
        "config": REPO_ROOT / "configs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_finetune_control.py",
        "work_dir": REPO_ROOT / "work_dirs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_finetune_control",
    },
    {
        "title": "LA-M2MRF-S",
        "config": REPO_ROOT / "configs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune.py",
        "work_dir": REPO_ROOT / "work_dirs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune",
    },
    {
        "title": "LA-M2MRF-W",
        "config": REPO_ROOT / "configs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune.py",
        "work_dir": REPO_ROOT / "work_dirs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune",
    },
    {
        "title": "LA-M2MRF-SW",
        "config": REPO_ROOT / "configs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune.py",
        "work_dir": REPO_ROOT / "work_dirs" / "la_m2mrf" / "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 qualitative comparison from real GT and model predictions."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help="Case ids without extension, e.g. IDRiD_60 IDRiD_68",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the figure PNG will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(FIG3_TEXT.keys()),
        default="zh",
        help="Output language for the figure title and legend labels.",
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


def case_paths(case_id: str) -> tuple[Path, Path]:
    image_path = DATA_ROOT / "image" / "test" / f"{case_id}.jpg"
    annotation_path = DATA_ROOT / "label" / "test" / "annotations" / f"{case_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")
    return image_path, annotation_path


def find_latest_checkpoint(work_dir: Path) -> Path:
    latest = work_dir / "latest.pth"
    if latest.is_file():
        return latest

    candidates = sorted(
        work_dir.glob("iter_*.pth"),
        key=lambda path: int(re.search(r"iter_(\\d+)\\.pth$", path.name).group(1)),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {work_dir}")
    return candidates[-1]


def decode_annotation_mask(mask_image: np.ndarray) -> np.ndarray:
    if mask_image.ndim == 2:
        return mask_image.astype(np.uint8)
    if mask_image.ndim != 3 or mask_image.shape[2] != 3:
        raise ValueError(f"Unsupported annotation shape: {mask_image.shape}")

    decoded = np.zeros(mask_image.shape[:2], dtype=np.uint8)
    bgr_palette = PALETTE[:, ::-1]
    for class_id, color in enumerate(bgr_palette):
        decoded[np.all(mask_image == color, axis=2)] = class_id
    return decoded


def result_to_segmentation(result) -> np.ndarray:
    if isinstance(result, (list, tuple)) and len(result) == 1:
        result = result[0]

    if isinstance(result, tuple):
        pred = result[0]
        use_sigmoid = bool(result[1]) if len(result) > 1 else False
        if use_sigmoid:
            binary = pred > 0.5
            seg = np.zeros(binary.shape[1:], dtype=np.uint8)
            for class_idx in range(binary.shape[0]):
                seg[binary[class_idx]] = class_idx + 1
            return seg
        return np.argmax(pred, axis=0).astype(np.uint8)

    if isinstance(result, list):
        result = result[0]
    return np.asarray(result, dtype=np.uint8)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = image_rgb.copy()
    for class_idx, color in DISPLAY_COLORS.items():
        overlay[mask == class_idx] = color

    blended = image_rgb.copy()
    valid_mask = mask > 0
    blended[valid_mask] = np.clip(
        (1.0 - alpha) * image_rgb[valid_mask] + alpha * overlay[valid_mask],
        0,
        255,
    ).astype(np.uint8)
    return blended


def resize_for_display(image_rgb: np.ndarray, height: int = DISPLAY_HEIGHT) -> np.ndarray:
    scale = height / float(image_rgb.shape[0])
    width = max(1, int(round(image_rgb.shape[1] * scale)))
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if mask.shape[:2] == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def load_case_visuals(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    image_path, annotation_path = case_paths(case_id)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    annotation_bgr = cv2.imread(str(annotation_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None or annotation_bgr is None:
        raise FileNotFoundError(f"Failed to read image or annotation for {case_id}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = decode_annotation_mask(annotation_bgr)
    return image_rgb, gt_mask


def infer_case_masks(case_ids: list[str], device: str) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {case_id: {} for case_id in case_ids}

    for experiment in EXPERIMENTS:
        checkpoint_path = find_latest_checkpoint(experiment["work_dir"])
        model = init_segmentor(str(experiment["config"]), str(checkpoint_path), device=device)
        try:
            for case_id in case_ids:
                image_path, _ = case_paths(case_id)
                result = inference_segmentor(model, str(image_path))
                predictions[case_id][experiment["title"]] = result_to_segmentation(result)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return predictions


def create_fig3(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    case_ids: list[str] | None = None,
    device: str | None = None,
    language: str = "zh",
) -> Path:
    configure_fonts()
    output_dir = Path(output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    if language not in FIG3_TEXT:
        raise ValueError(f"Unsupported language: {language}")
    text = FIG3_TEXT[language]

    case_ids = case_ids or DEFAULT_CASES
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    predictions = infer_case_masks(case_ids, device=device)

    columns = ["Original Image", "Ground Truth"] + [item["title"] for item in EXPERIMENTS]
    fig, axes = plt.subplots(len(case_ids), len(columns), figsize=(18, 5 * len(case_ids)))
    if len(case_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, case_id in enumerate(case_ids):
        image_rgb, gt_mask = load_case_visuals(case_id)
        gt_overlay = overlay_mask(image_rgb, gt_mask)

        row_images = [image_rgb, gt_overlay]
        for experiment in EXPERIMENTS:
            pred_mask = resize_mask(predictions[case_id][experiment["title"]], image_rgb.shape[:2])
            row_images.append(overlay_mask(image_rgb, pred_mask))

        for col_idx, (title, panel_rgb) in enumerate(zip(columns, row_images)):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(title, fontsize=14, pad=10)

            if col_idx == 0:
                ax.set_ylabel(
                    f"Case {row_idx + 1}\n({case_id})",
                    fontsize=14,
                    labelpad=20,
                    rotation=0,
                    ha="right",
                    va="center",
                )

            ax.imshow(resize_for_display(panel_rgb))
            ax.set_xticks([])
            ax.set_yticks([])

    legend_handles = [
        mpatches.Patch(color="yellow", label=text["legend_ex"]),
        mpatches.Patch(color="red", label=text["legend_he"]),
        mpatches.Patch(color="blue", label=text["legend_se"]),
        mpatches.Patch(color="lime", label=text["legend_ma"]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.suptitle(text["figure_title"], fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_path = output_dir / text["output_name"]
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = create_fig3(
        output_dir=args.output_dir,
        case_ids=args.cases,
        device=args.device,
        language=args.language,
    )
    print(f"Saved Figure 3 to: {output_path}")


if __name__ == "__main__":
    main()
