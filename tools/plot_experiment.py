#!/usr/bin/env python3
"""Parse mmseg logs and plot training/validation metrics using iteration as x-axis.

Usage: python tools/plot_experiment.py <experiment_dir>

Produces: plots/loss_iter.png and plots/classification_report_iter.png
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_logs(root: Path):
    log_json = None
    log_txt = None
    for p in root.iterdir():
        if p.name.endswith('.log.json'):
            log_json = p
        if p.name.endswith('.log') and not p.name.endswith('.log.json'):
            log_txt = p
    if not log_txt and not log_json:
        raise SystemExit('no logs found in ' + str(root))

    epoch_losses = defaultdict(list)
    raw_iters = []
    if log_json and log_json.exists():
        with log_json.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get('mode') != 'train' or 'loss' not in obj:
                    continue
                epoch = int(obj.get('epoch', 0))
                it = int(obj.get('iter', 0))
                loss = float(obj.get('loss', 0.0))
                epoch_losses[epoch].append(loss)
                raw_iters.append((it, epoch, loss))

    eval_points = []
    class_reports = []
    if log_txt and log_txt.exists():
        lines = log_txt.read_text(encoding='utf-8', errors='ignore').splitlines()
        for idx, line in enumerate(lines):
            m = re.search(r'Iter \[(\d+)/\d+\].*mIoU: ([0-9.]+), mF1: ([0-9.]+), mPPV: ([0-9.]+), mS: ([0-9.]+), mAUPR: ([0-9.]+)', line)
            if m:
                eval_points.append({
                    'iter': int(m.group(1)),
                    'mIoU': float(m.group(2)),
                    'mF1': float(m.group(3)),
                    'mPPV': float(m.group(4)),
                    'mS': float(m.group(5)),
                    'mAUPR': float(m.group(6)),
                })
            if 'per class results:' in line:
                table = {}
                j = idx + 2
                while j < len(lines) and 'Summary:' not in lines[j]:
                    row = lines[j].strip()
                    if row and not row.startswith('Class'):
                        parts = row.split()
                        if len(parts) >= 6:
                            def to_float(x):
                                return float('nan') if x.lower() == 'nan' else float(x)
                            table[parts[0]] = {
                                'IoU': to_float(parts[1]),
                                'F1': to_float(parts[2]),
                                'PPV': to_float(parts[3]),
                                'S': to_float(parts[4]),
                                'AUPR': to_float(parts[5]),
                            }
                    j += 1
                back = idx
                iter_num = None
                while back >= 0:
                    mm = re.search(r'Iter \[(\d+)/\d+\]', lines[back])
                    if mm:
                        iter_num = int(mm.group(1))
                        break
                    back -= 1
                class_reports.append((iter_num, table))

    return epoch_losses, raw_iters, eval_points, class_reports


def plot_iter_plots(root: Path, epoch_losses, raw_iters, eval_points, class_reports):
    out = root / 'plots'
    out.mkdir(exist_ok=True)

    # Loss per epoch but x-axis use iteration (mean iteration per epoch)
    epochs = sorted(epoch_losses)
    mean_losses = []
    epoch_iters = []
    for e in epochs:
        vals = epoch_losses[e]
        mean_losses.append(sum(vals) / len(vals))
        # estimate median iteration for epoch from raw_iters
        it_vals = [it for it, ep, _ in raw_iters if ep == e]
        epoch_iters.append(int(sum(it_vals) / len(it_vals)) if it_vals else e * 100)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(epoch_iters, mean_losses, color='#c75c2a', linewidth=1.8, label='Epoch mean loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss by Iteration (epoch-mean)')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    loss_path = out / 'loss_iter.png'
    fig.savefig(loss_path, bbox_inches='tight')
    plt.close(fig)

    # Validation metrics plotted against iteration
    report_path = None
    if eval_points:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
        metrics = [('mIoU', '#1f77b4'), ('mF1', '#ff7f0e'), ('mPPV', '#2ca02c'), ('mS', '#9467bd'), ('mAUPR', '#d62728')]
        iters = [p['iter'] for p in eval_points]
        for metric, color in metrics:
            axes[0].plot(iters, [p[metric] for p in eval_points], marker='o', linewidth=2, label=metric, color=color)
        axes[0].set_title('Validation Summary Metrics by Iteration')
        axes[0].set_ylabel('Score')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(ncol=3, frameon=False)

        # per-class F1
        for cls, color in zip(['EX', 'HE', 'SE', 'MA'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
            axes[1].plot([it for it, _ in class_reports], [table.get(cls, {}).get('F1', float('nan')) for _, table in class_reports], marker='o', linewidth=2, label=f'{cls} F1', color=color)
        axes[1].set_title('Per-class F1 in Classification Report')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('F1 (%)')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(ncol=2, frameon=False)
        fig.tight_layout()
        report_path = out / 'classification_report_iter.png'
        fig.savefig(report_path, bbox_inches='tight')
        plt.close(fig)

    return loss_path, report_path


def main():
    if len(sys.argv) < 2:
        print('usage: python tools/plot_experiment.py <experiment_dir>')
        raise SystemExit(1)
    root = Path(sys.argv[1])
    epoch_losses, raw_iters, eval_points, class_reports = parse_logs(root)
    loss_path, report_path = plot_iter_plots(root, epoch_losses, raw_iters, eval_points, class_reports)
    print('Wrote:', loss_path)
    if report_path:
        print('Wrote:', report_path)


if __name__ == '__main__':
    main()
