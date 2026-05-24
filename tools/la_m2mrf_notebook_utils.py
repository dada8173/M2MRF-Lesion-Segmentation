"""Utilities for the LA-M2MRF reproduction notebook workflow."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import mmcv
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional import in some environments
    Image = None
    ImageDraw = None
    ImageFont = None


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / 'work_dirs' / 'la_m2mrf' / 'notebook_artifacts'
OFFICIAL_CHECKPOINT_NAME = 'fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth'
PALETTE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
], dtype=np.uint8)
CLASS_NAMES = ['bg', 'EX', 'HE', 'SE', 'MA']
CLASS_WEIGHTS = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0}
IDRID_IMAGE_GLOB = ('*.jpg', '*.jpeg', '*.png')
DEFAULT_SAMPLE_IMAGE = 'IDRiD_36.jpg'
CJK_FONT_CANDIDATES = [
    'Noto Sans CJK TC',
    'Noto Sans CJK SC',
    'Microsoft JhengHei',
    'Microsoft YaHei',
    'PingFang TC',
    'Source Han Sans TW',
]
CJK_FONT_PATH_HINTS = [
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
    Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'),
    Path('/mnt/c/Windows/Fonts/msjh.ttc'),
    Path('/mnt/c/Windows/Fonts/msjhbd.ttc'),
    Path('/mnt/c/Windows/Fonts/msyh.ttc'),
    Path('/mnt/c/Windows/Fonts/mingliu.ttc'),
]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    config_path: str
    work_dir: str
    load_from_official: bool = False

    @property
    def config(self) -> Path:
        return REPO_ROOT / self.config_path

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / self.work_dir

    @property
    def cache_path(self) -> Path:
        return ARTIFACT_ROOT / 'metrics_cache' / f'{self.key}.json'


BASELINE_SCRATCH_SPEC = ExperimentSpec(
    key='baseline_scratch_bs3_fp16',
    label='Baseline scratch / 基線從零訓練',
    config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_official_scratch_bs3_fp16.py',
    work_dir='work_dirs/la_m2mrf/idrid_official_scratch_bs3_fp16',
)

FINETUNE_EXPERIMENTS = [
    ExperimentSpec(
        key='finetune_baseline',
        label='Baseline / 基線',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_finetune_control.py',
        work_dir='work_dirs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_finetune_control',
        load_from_official=True,
    ),
    ExperimentSpec(
        key='finetune_la_sampler',
        label='A / 病灶感知採樣',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune.py',
        work_dir='work_dirs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune',
        load_from_official=True,
    ),
    ExperimentSpec(
        key='finetune_weighted_dice',
        label='B / 加權 Dice',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune.py',
        work_dir='work_dirs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune',
        load_from_official=True,
    ),
    ExperimentSpec(
        key='finetune_la_sampler_weighted_dice',
        label='A+B / 組合方法',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune.py',
        work_dir='work_dirs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune',
        load_from_official=True,
    ),
]

SCRATCH_EXPERIMENTS = [
    ExperimentSpec(
        key='scratch_baseline',
        label='Baseline / 基線',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_official_scratch_bs3_fp16.py',
        work_dir='work_dirs/la_m2mrf/idrid_official_scratch_bs3_fp16',
    ),
    ExperimentSpec(
        key='scratch_la_sampler',
        label='A / 病灶感知採樣',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_la_sampler_scratch_bs3_fp16.py',
        work_dir='work_dirs/la_m2mrf/idrid_la_sampler_scratch_bs3_fp16',
    ),
    ExperimentSpec(
        key='scratch_weighted_dice',
        label='B / 加權 Dice',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_weighted_dice_scratch_bs3_fp16.py',
        work_dir='work_dirs/la_m2mrf/idrid_weighted_dice_scratch_bs3_fp16',
    ),
    ExperimentSpec(
        key='scratch_la_sampler_weighted_dice',
        label='A+B / 組合方法',
        config_path='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_la_sampler_weighted_dice_scratch_bs3_fp16.py',
        work_dir='work_dirs/la_m2mrf/idrid_la_sampler_weighted_dice_scratch_bs3_fp16',
    ),
]


def ensure_artifact_root() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / 'metrics_cache').mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


def _find_cjk_font_path() -> Optional[Path]:
    for font_path in CJK_FONT_PATH_HINTS:
        if font_path.is_file():
            return font_path

    for font_file in font_manager.findSystemFonts():
        lower_name = Path(font_file).name.lower()
        if any(keyword in lower_name for keyword in ['notosanscjk', 'msjh', 'msyh', 'mingliu', 'simsun']):
            return Path(font_file)
    return None


def configure_plot_fonts() -> Optional[Path]:
    font_path = _find_cjk_font_path()
    if font_path is None:
        return None

    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['font.family'] = [font_name, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return font_path


configure_plot_fonts()


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def dataset_root() -> Path:
    return (REPO_ROOT / '../data/IDRID').resolve()


def dataset_exists() -> bool:
    return dataset_root().is_dir()


def assert_dataset_ready() -> Path:
    root = dataset_root()
    if not root.is_dir():
        raise FileNotFoundError(
            'IDRiD data not found at ../data/IDRID. Please follow docs/DATASET_SETUP_IDRID.md.')
    return root


def environment_dependency_markdown() -> str:
    font_path = _find_cjk_font_path()
    return '\n'.join([
        '- Notebook kernel: Python 3.10+ is recommended for this workspace.',
        '- Repo compatibility target: PyTorch 1.6.0 / MMCV 1.2.0 / MMSeg 0.8.0.',
        '- Workspace-verified modern base: Python 3.10.20 with CUDA-enabled PyTorch 2.11.0+cu130.',
        '- Required utility packages for this notebook: `opencv-python`, `scikit-learn`, `matplotlib`, `tensorboard`, `tensorboardX`, `ipykernel`, `pandas`.',
        '- Dataset path expected by configs: `../data/IDRID` relative to repo root.',
        '- Official checkpoint file expected: `fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth`.',
        '- Chinese plotting font: `{}`'.format(font_path if font_path is not None else 'not found; English fallback will be used'),
    ])


def install_commands() -> List[str]:
    return [
        'source ~/miniconda3/etc/profile.d/conda.sh',
        'conda activate la-m2mrf',
        'python -m pip install -U pip',
        'python -m pip install -r requirements.txt',
        'python -m pip install opencv-python scikit-learn pandas ipykernel tensorboard tensorboardX',
        'python -m pip install -e .',
    ]


def format_commands(commands: Sequence[str]) -> str:
    return '\n'.join(commands)


def run_command(command: Sequence[str], cwd: Optional[Path] = None) -> str:
    cwd = cwd or REPO_ROOT
    print('$', ' '.join(command))
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            'Command failed with exit code {}:\n{}'.format(proc.returncode, proc.stdout))
    return proc.stdout


def _official_checkpoint_candidates() -> Iterable[Path]:
    env_path = os.environ.get('LA_M2MRF_OFFICIAL_CKPT')
    if env_path:
        yield Path(env_path).expanduser()

    base_candidates = [
        REPO_ROOT / 'checkpoints' / OFFICIAL_CHECKPOINT_NAME,
        REPO_ROOT / OFFICIAL_CHECKPOINT_NAME,
        Path('/mnt/c/Users/dachen/Downloads') / OFFICIAL_CHECKPOINT_NAME,
        Path.home() / 'Downloads' / OFFICIAL_CHECKPOINT_NAME,
    ]
    for candidate in base_candidates:
        yield candidate

    for user_dir in Path('/mnt/c/Users').glob('*/Downloads'):
        yield user_dir / OFFICIAL_CHECKPOINT_NAME


def find_official_checkpoint() -> Path:
    for candidate in _official_checkpoint_candidates():
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        'Could not find the official checkpoint {}. Set LA_M2MRF_OFFICIAL_CKPT if needed.'.format(
            OFFICIAL_CHECKPOINT_NAME))


def find_latest_checkpoint(work_dir: Path) -> Optional[Path]:
    if not work_dir.is_dir():
        return None
    latest = work_dir / 'latest.pth'
    if latest.is_file():
        return latest
    candidates = sorted(
        work_dir.glob('iter_*.pth'),
        key=lambda path: int(re.search(r'iter_(\d+)\.pth$', path.name).group(1))
        if re.search(r'iter_(\d+)\.pth$', path.name) else -1)
    return candidates[-1] if candidates else None


def iter_from_checkpoint_name(path: Path) -> Optional[int]:
    match = re.search(r'iter_(\d+)\.pth$', path.name)
    return int(match.group(1)) if match else None


def latest_log_path(work_dir: Path) -> Optional[Path]:
    logs = sorted(work_dir.glob('*.log'))
    return logs[-1] if logs else None


def latest_log_json_path(work_dir: Path) -> Optional[Path]:
    logs = sorted(work_dir.glob('*.log.json'))
    return logs[-1] if logs else None


def ensure_training(
        spec: ExperimentSpec,
        train_if_missing: bool = True,
        force_train: bool = False,
        seed: int = 0,
        gpus: int = 1,
        extra_options: Optional[Sequence[str]] = None) -> Path:
    checkpoint = find_latest_checkpoint(spec.workdir)
    if checkpoint and not force_train:
        print('Reusing checkpoint for {}: {}'.format(spec.label, checkpoint))
        return checkpoint

    if not train_if_missing:
        raise FileNotFoundError(
            'Checkpoint missing for {} under {}.'.format(spec.label, spec.workdir))

    assert_dataset_ready()
    spec.workdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        'tools/train.py',
        spec.config_path,
        '--work-dir',
        spec.work_dir,
        '--gpus',
        str(gpus),
        '--seed',
        str(seed),
        '--deterministic',
    ]
    if spec.load_from_official:
        command.extend(['--load-from', str(find_official_checkpoint())])
    if extra_options:
        command.extend(extra_options)
    run_command(command)
    checkpoint = find_latest_checkpoint(spec.workdir)
    if checkpoint is None:
        raise FileNotFoundError(
            'Training finished but no checkpoint was found in {}'.format(spec.workdir))
    return checkpoint


def _parse_log_json(log_json: Path) -> Tuple[List[Tuple[int, float]], Dict[int, List[float]]]:
    iter_losses = []
    epoch_buckets: Dict[int, List[float]] = {}
    if log_json is None or not log_json.is_file():
        return iter_losses, epoch_buckets

    with log_json.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if obj.get('mode') != 'train' or 'loss' not in obj:
                continue
            iteration = int(obj.get('iter', 0))
            loss = float(obj.get('loss', 0.0))
            epoch = int(obj.get('epoch', 0))
            iter_losses.append((iteration, loss))
            epoch_buckets.setdefault(epoch, []).append(loss)
    return iter_losses, epoch_buckets


def _parse_log_text(log_path: Path) -> Tuple[List[dict], List[Tuple[Optional[int], Dict[str, dict]]]]:
    eval_points: List[dict] = []
    class_reports: List[Tuple[Optional[int], Dict[str, dict]]] = []
    if log_path is None or not log_path.is_file():
        return eval_points, class_reports

    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    metric_pattern = re.compile(
        r'Iter \[(\d+)/\d+\].*mIoU: ([0-9.]+), mF1: ([0-9.]+), '
        r'mPPV: ([0-9.]+), mS: ([0-9.]+), mAUPR: ([0-9.]+)')
    iter_pattern = re.compile(r'Iter \[(\d+)/\d+\]')

    for idx, line in enumerate(lines):
        match = metric_pattern.search(line)
        if match:
            eval_points.append({
                'iter': int(match.group(1)),
                'mIoU': float(match.group(2)),
                'mF1': float(match.group(3)),
                'mPPV': float(match.group(4)),
                'mS': float(match.group(5)),
                'mAUPR': float(match.group(6)),
            })
        if 'per class results:' in line:
            table: Dict[str, dict] = {}
            j = idx + 2
            while j < len(lines) and 'Summary:' not in lines[j]:
                row = lines[j].strip()
                if row and not row.startswith('Class'):
                    parts = row.split()
                    if len(parts) >= 6:
                        table[parts[0]] = {
                            'IoU': float(parts[1]) if parts[1].lower() != 'nan' else math.nan,
                            'F1': float(parts[2]) if parts[2].lower() != 'nan' else math.nan,
                            'PPV': float(parts[3]) if parts[3].lower() != 'nan' else math.nan,
                            'S': float(parts[4]) if parts[4].lower() != 'nan' else math.nan,
                            'AUPR': float(parts[5]) if parts[5].lower() != 'nan' else math.nan,
                        }
                j += 1

            iter_num: Optional[int] = None
            back = idx
            while back >= 0:
                iter_match = iter_pattern.search(lines[back])
                if iter_match:
                    iter_num = int(iter_match.group(1))
                    break
                back -= 1
            class_reports.append((iter_num, table))
    return eval_points, class_reports


def ensure_loss_plots(work_dir: Path) -> Dict[str, Path]:
    configure_plot_fonts()
    log_json = latest_log_json_path(work_dir)
    log_path = latest_log_path(work_dir)
    iter_losses, epoch_buckets = _parse_log_json(log_json)
    eval_points, class_reports = _parse_log_text(log_path)
    if not iter_losses and not epoch_buckets:
        raise FileNotFoundError('No training loss logs found in {}'.format(work_dir))

    out_dir = work_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_iter_path = out_dir / 'loss_iter.png'
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.plot(
        [item[0] for item in iter_losses],
        [item[1] for item in iter_losses],
        color='#bf5b17',
        linewidth=1.2)
    ax.set_title('Training Loss by Iteration')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_iter_path, bbox_inches='tight')
    plt.close(fig)

    loss_epoch_path = out_dir / 'loss_epoch.png'
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    epochs = sorted(epoch_buckets)
    ax.plot(
        epochs,
        [float(np.mean(epoch_buckets[epoch])) for epoch in epochs],
        marker='o',
        color='#2a6f97',
        linewidth=1.6)
    ax.set_title('Training Loss by Epoch Mean')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_epoch_path, bbox_inches='tight')
    plt.close(fig)

    classification_iter_path = out_dir / 'classification_report_iter.png'
    classification_epoch_path = out_dir / 'classification_report_epoch.png'
    if eval_points:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
        colors = {
            'mIoU': '#1d3557',
            'mF1': '#e76f51',
            'mPPV': '#2a9d8f',
            'mS': '#8d99ae',
            'mAUPR': '#d62828',
        }
        iters = [row['iter'] for row in eval_points]
        for metric, color in colors.items():
            axes[0].plot(iters, [row[metric] for row in eval_points], marker='o', linewidth=2, label=metric, color=color)
        axes[0].set_title('Validation Summary Metrics')
        axes[0].set_ylabel('Score')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False, ncol=3)

        for cls_name, color in zip(['EX', 'HE', 'SE', 'MA'], ['#d7301f', '#1b9e77', '#7570b3', '#e6ab02']):
            axes[1].plot(
                [entry[0] for entry in class_reports],
                [entry[1].get(cls_name, {}).get('IoU', math.nan) for entry in class_reports],
                marker='o',
                linewidth=2,
                label='{} IoU'.format(cls_name),
                color=color)
        axes[1].set_title('Per-class IoU')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('IoU (%)')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(classification_iter_path, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
        epochs_for_eval = list(range(1, len(eval_points) + 1))
        for metric, color in colors.items():
            axes[0].plot(epochs_for_eval, [row[metric] for row in eval_points], marker='o', linewidth=2, label=metric, color=color)
        axes[0].set_title('Validation Summary Metrics')
        axes[0].set_ylabel('Score')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False, ncol=3)
        for cls_name, color in zip(['EX', 'HE', 'SE', 'MA'], ['#d7301f', '#1b9e77', '#7570b3', '#e6ab02']):
            axes[1].plot(
                epochs_for_eval[:len(class_reports)],
                [entry[1].get(cls_name, {}).get('IoU', math.nan) for entry in class_reports],
                marker='o',
                linewidth=2,
                label='{} IoU'.format(cls_name),
                color=color)
        axes[1].set_title('Per-class IoU')
        axes[1].set_xlabel('Evaluation Index')
        axes[1].set_ylabel('IoU (%)')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(classification_epoch_path, bbox_inches='tight')
        plt.close(fig)

    return {
        'loss_iter': loss_iter_path,
        'loss_epoch': loss_epoch_path,
        'classification_iter': classification_iter_path,
        'classification_epoch': classification_epoch_path,
    }


def parse_metrics_from_text(text: str) -> dict:
    lines = text.splitlines()
    per_class: Dict[str, dict] = {}
    summary = {}

    for idx, line in enumerate(lines):
        if 'per class results:' in line:
            j = idx + 2
            while j < len(lines) and 'Summary:' not in lines[j]:
                row = lines[j].strip()
                if row and not row.startswith('Class'):
                    parts = row.split()
                    if len(parts) >= 6:
                        per_class[parts[0]] = {
                            'IoU': float(parts[1]) if parts[1].lower() != 'nan' else math.nan,
                            'F1': float(parts[2]) if parts[2].lower() != 'nan' else math.nan,
                            'PPV': float(parts[3]) if parts[3].lower() != 'nan' else math.nan,
                            'S': float(parts[4]) if parts[4].lower() != 'nan' else math.nan,
                            'AUPR': float(parts[5]) if parts[5].lower() != 'nan' else math.nan,
                        }
                j += 1
        if 'Summary:' in line:
            j = idx + 3
            if j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 6:
                    summary = {
                        'mIoU': float(parts[1]),
                        'mF1': float(parts[2]),
                        'mPPV': float(parts[3]),
                        'mS': float(parts[4]),
                        'mAUPR': float(parts[5]),
                    }
            break

    summary['per_class'] = per_class
    return summary


def latest_training_metrics(work_dir: Path) -> Optional[dict]:
    log_path = latest_log_path(work_dir)
    if log_path is None:
        return None
    eval_points, class_reports = _parse_log_text(log_path)
    if not eval_points:
        return None
    summary = dict(eval_points[-1])
    if class_reports:
        summary['per_class'] = class_reports[-1][1]
    return summary


def evaluate_checkpoint(
        label: str,
        config_path: Path,
        checkpoint_path: Path,
        cache_key: str,
        force_eval: bool = False) -> dict:
    ensure_artifact_root()
    cache_path = ARTIFACT_ROOT / 'metrics_cache' / f'{cache_key}.json'
    if cache_path.is_file() and not force_eval:
        return json.loads(cache_path.read_text(encoding='utf-8'))

    assert_dataset_ready()
    command = [
        sys.executable,
        'tools/test.py',
        str(config_path),
        str(checkpoint_path),
        '--eval',
        'mIoU',
    ]
    stdout = run_command(command)
    metrics = parse_metrics_from_text(stdout)
    metrics.update({
        'label': label,
        'config_path': str(config_path),
        'checkpoint_path': str(checkpoint_path),
        'source': 'tools/test.py',
    })
    cache_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def collect_metrics(
        spec: ExperimentSpec,
        checkpoint_path: Optional[Path] = None,
        force_eval: bool = False) -> dict:
    checkpoint_path = checkpoint_path or find_latest_checkpoint(spec.workdir)
    if checkpoint_path is None:
        raise FileNotFoundError('No checkpoint found for {}'.format(spec.label))

    cached_from_log = latest_training_metrics(spec.workdir)
    if cached_from_log is not None and not force_eval:
        metrics = dict(cached_from_log)
        metrics.update({
            'label': spec.label,
            'config_path': str(spec.config),
            'checkpoint_path': str(checkpoint_path),
            'source': 'train_log',
        })
        spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
        spec.cache_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        return metrics

    return evaluate_checkpoint(
        label=spec.label,
        config_path=spec.config,
        checkpoint_path=checkpoint_path,
        cache_key=spec.key,
        force_eval=force_eval)


def summarize_rows(metrics_rows: Sequence[dict]) -> List[dict]:
    rows = []
    for row in metrics_rows:
        per_class = row.get('per_class', {})
        rows.append({
            'Method': row['label'],
            'mIoU': row.get('mIoU'),
            'mF1': row.get('mF1'),
            'mAUPR': row.get('mAUPR'),
            'EX IoU': per_class.get('EX', {}).get('IoU'),
            'HE IoU': per_class.get('HE', {}).get('IoU'),
            'SE IoU': per_class.get('SE', {}).get('IoU'),
            'MA IoU': per_class.get('MA', {}).get('IoU'),
            'Checkpoint': row.get('checkpoint_path'),
            'Source': row.get('source'),
        })
    return rows


def _find_sample_image(root: Path) -> Path:
    preferred = root / 'image' / 'test' / DEFAULT_SAMPLE_IMAGE
    if preferred.is_file():
        return preferred
    for split in ['test', 'train']:
        image_dir = root / 'image' / split
        if not image_dir.is_dir():
            continue
        for pattern in IDRID_IMAGE_GLOB:
            candidates = sorted(image_dir.glob(pattern))
            if candidates:
                return candidates[0]
    raise FileNotFoundError('No sample image found under {}'.format(root / 'image'))


def _find_sample_annotation(root: Path, image_path: Path) -> Path:
    split = image_path.parent.name
    annotation_path = root / 'label' / split / 'annotations' / image_path.with_suffix('.png').name
    if not annotation_path.is_file():
        raise FileNotFoundError('Annotation not found for sample image: {}'.format(annotation_path))
    return annotation_path


def default_sample_paths() -> Tuple[Path, Path]:
    root = assert_dataset_ready()
    image_path = _find_sample_image(root)
    annotation_path = _find_sample_annotation(root, image_path)
    return image_path, annotation_path


def _to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_for_panel(image: np.ndarray, height: int = 320) -> np.ndarray:
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _add_title_strip(image_rgb: np.ndarray, title: str) -> np.ndarray:
    title_strip = np.full((54, image_rgb.shape[1], 3), 255, dtype=np.uint8)
    font_path = _find_cjk_font_path()

    if Image is not None and ImageDraw is not None and ImageFont is not None and font_path is not None:
        canvas = Image.fromarray(title_strip)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(font_path), 24)
        draw.text((12, 12), title, fill=(0, 0, 0), font=font)
        title_strip = np.array(canvas)
    else:
        cv2.putText(
            title_strip,
            title,
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA)
    return np.vstack([title_strip, image_rgb])


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = PALETTE[mask]
    overlay = image_rgb.copy()
    lesion = mask > 0
    overlay[lesion] = np.clip(
        (1.0 - alpha) * overlay[lesion] + alpha * colored[lesion],
        0,
        255).astype(np.uint8)
    return overlay


def _build_weight_overlay(mask: np.ndarray) -> np.ndarray:
    weights = np.zeros(mask.shape, dtype=np.float32)
    for cls_id, cls_weight in CLASS_WEIGHTS.items():
        weights[mask == cls_id] = cls_weight
    if weights.max() <= 0:
        return np.zeros(mask.shape + (3,), dtype=np.uint8)
    normalized = ((weights - weights.min()) / max(weights.max() - weights.min(), 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def _decode_annotation_mask(mask_image: np.ndarray) -> np.ndarray:
    if mask_image.ndim == 2:
        return mask_image.astype(np.uint8)
    if mask_image.ndim != 3 or mask_image.shape[2] != 3:
        raise ValueError('Unsupported annotation shape: {}'.format(mask_image.shape))

    bgr_palette = PALETTE[:, ::-1]
    decoded = np.zeros(mask_image.shape[:2], dtype=np.uint8)
    for class_id, color in enumerate(bgr_palette):
        decoded[np.all(mask_image == color, axis=2)] = class_id
    return decoded


def build_method_preview(output_path: Path, seed: int = 0) -> Path:
    from mmseg.datasets.pipelines.la_m2mrf_transforms import LesionAwareRandomCrop

    image_path, annotation_path = default_sample_paths()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask_image = cv2.imread(str(annotation_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None or mask_image is None:
        raise FileNotFoundError('Failed to load sample image or mask.')
    mask = _decode_annotation_mask(mask_image)

    np.random.seed(seed)
    cropper = LesionAwareRandomCrop(
        crop_size=(960, 1440),
        lesion_prob=0.7,
        small_lesion_prob=0.4,
        target_classes=(1, 2, 3, 4),
        priority_classes=(3, 4),
        cat_max_ratio=0.75,
        num_retry=10)

    random_bbox = cropper.get_random_crop_bbox(image_bgr)
    lesion_bbox = cropper.get_lesion_crop_bbox(mask, cropper._pick_candidate_classes(mask))
    if lesion_bbox is None:
        lesion_bbox = random_bbox

    base_rgb = _to_rgb(image_bgr)
    a_rgb = base_rgb.copy()
    b_rgb = _overlay_mask(base_rgb, mask)
    ab_rgb = _overlay_mask(base_rgb, mask)

    for canvas, bbox, color in [
            (a_rgb, lesion_bbox, (255, 99, 71)),
            (ab_rgb, lesion_bbox, (255, 99, 71)),
    ]:
        y1, y2, x1, x2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 12)

    baseline_rgb = base_rgb.copy()
    y1, y2, x1, x2 = random_bbox
    cv2.rectangle(baseline_rgb, (x1, y1), (x2, y2), (70, 130, 180), 12)

    weight_rgb = _build_weight_overlay(mask)
    b_rgb = np.clip(0.55 * b_rgb + 0.45 * weight_rgb, 0, 255).astype(np.uint8)
    ab_rgb = np.clip(0.55 * ab_rgb + 0.45 * weight_rgb, 0, 255).astype(np.uint8)

    panels = [
        _add_title_strip(_resize_for_panel(base_rgb), 'Original / 原圖'),
        _add_title_strip(_resize_for_panel(a_rgb), 'A / 病灶感知採樣'),
        _add_title_strip(_resize_for_panel(b_rgb), 'B / 加權 Dice'),
        _add_title_strip(_resize_for_panel(ab_rgb), 'A+B / 採樣 + 加權損失'),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
    return output_path


def _show_result_image(config_path: Path, checkpoint_path: Path, image_path: Path) -> np.ndarray:
    import torch
    from mmseg.apis import inference_segmentor, init_segmentor

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_segmentor(str(config_path), str(checkpoint_path), device=device)
    result = inference_segmentor(model, str(image_path))
    image_bgr = mmcv.imread(str(image_path))
    seg = _result_to_segmentation(result)
    painted = _overlay_segmentation_bgr(image_bgr, seg)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _to_rgb(painted)


def _result_to_segmentation(result) -> np.ndarray:
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


def _overlay_segmentation_bgr(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE[:, ::-1]):
        color_seg[seg == class_id] = color
    blended = image_bgr.astype(np.float32) * 0.5 + color_seg.astype(np.float32) * 0.5
    return blended.astype(np.uint8)


def build_prediction_panel(
        items: Sequence[Tuple[str, Path, Path]],
        output_path: Path,
        sample_image: Optional[Path] = None) -> Path:
    sample_image = sample_image or default_sample_paths()[0]
    image_bgr = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError('Sample image not found: {}'.format(sample_image))

    panels = [_add_title_strip(_resize_for_panel(_to_rgb(image_bgr)), 'Original / 原圖')]
    for label, config_path, checkpoint_path in items:
        painted = _show_result_image(config_path, checkpoint_path, sample_image)
        panels.append(_add_title_strip(_resize_for_panel(painted), label))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
    return output_path


def build_loss_grid(
        specs: Sequence[ExperimentSpec],
        output_path: Path,
        title: str) -> Path:
    configure_plot_fonts()
    figure_paths = []
    for spec in specs:
        plots = ensure_loss_plots(spec.workdir)
        figure_paths.append((spec.label, plots['loss_iter']))

    cols = 2
    rows = int(math.ceil(len(figure_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows), dpi=160)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis('off')

    for ax, (label, path) in zip(axes.ravel(), figure_paths):
        image = plt.imread(str(path))
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.axis('off')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path
