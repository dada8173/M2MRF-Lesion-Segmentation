#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DISCORD_ROOT = Path('/home/dachen/onevision_next')
DISCORD_SENDER = DISCORD_ROOT / 'dachen_discord_webhook_send.py'
DEFAULT_PYTHON = '/home/dachen/miniconda3/envs/la-m2mrf/bin/python'
DEFAULT_WORKERS = '0'
WAIT_INTERVAL_SECONDS = 900
RESULTS_DIR = REPO_ROOT / 'work_dirs/la_m2mrf/results'
RESULTS_MD = RESULTS_DIR / 'idrid_comparison_table.md'
RESULTS_CSV = RESULTS_DIR / 'idrid_comparison_table.csv'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S,%f'


@dataclass(frozen=True)
class Experiment:
    name: str
    subject: str
    config: str
    work_dir: str
    method_label: str


EXPERIMENTS = [
    Experiment(
        name='idrid_full',
        subject='LA-M2MRF full 完成',
        config='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_full.py',
        work_dir='work_dirs/la_m2mrf/idrid_full',
        method_label='LA-M2MRF full',
    ),
    Experiment(
        name='idrid_la_sampler',
        subject='LA sampler ablation 完成',
        config='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_la_sampler.py',
        work_dir='work_dirs/la_m2mrf/idrid_la_sampler',
        method_label='LA sampler only',
    ),
    Experiment(
        name='idrid_weighted_dice',
        subject='Weighted dice ablation 完成',
        config='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_weighted_dice.py',
        work_dir='work_dirs/la_m2mrf/idrid_weighted_dice',
        method_label='Weighted dice only',
    ),
    Experiment(
        name='idrid_fundus_enhance',
        subject='Fundus enhance ablation 完成',
        config='configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_fundus_enhance.py',
        work_dir='work_dirs/la_m2mrf/idrid_fundus_enhance',
        method_label='Fundus enhance only',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Queue LA-M2MRF experiments and notify Discord on completion.')
    parser.add_argument('--wait-pid', type=int, help='Existing training PID to wait for before starting the queue.')
    parser.add_argument('--wait-name', default='idrid_baseline_copy', help='Display name for the existing run.')
    parser.add_argument('--wait-work-dir', default='work_dirs/la_m2mrf/idrid_baseline_copy',
                        help='Work dir of the existing run to summarize when it finishes.')
    parser.add_argument('--python-bin', default=DEFAULT_PYTHON, help='Python executable for training runs.')
    parser.add_argument('--seed', default='0', help='Training seed.')
    parser.add_argument('--workers-per-gpu', default=DEFAULT_WORKERS, help='Override data.workers_per_gpu.')
    parser.add_argument('--route', default='summary', choices=['default', 'full', 'summary'],
                        help='Discord route for notifications.')
    parser.add_argument('--sleep-seconds', type=int, default=WAIT_INTERVAL_SECONDS,
                        help='Polling interval when waiting on an external PID.')
    return parser.parse_args()


def latest_log_file(work_dir: Path) -> Optional[Path]:
    logs = sorted(work_dir.glob('*.log'))
    return logs[-1] if logs else None


def parse_duration(log_text: str) -> Optional[float]:
    timestamps = re.findall(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', log_text, flags=re.MULTILINE)
    if len(timestamps) < 2:
        return None
    start = datetime.strptime(timestamps[0], TIME_FORMAT)
    end = datetime.strptime(timestamps[-1], TIME_FORMAT)
    return max((end - start).total_seconds(), 0.0)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return 'unknown'
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def parse_summary(log_path: Path) -> dict:
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    summary_matches = re.findall(
        r'global\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',
        text)
    metric_names = ['mIoU', 'mF1', 'mPPV', 'mS', 'mAUPR']
    metrics = {}
    if summary_matches:
        last = summary_matches[-1]
        metrics = {name: value for name, value in zip(metric_names, last)}

    class_matches = re.findall(
        r'^(EX|HE|SE|MA)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$',
        text,
        flags=re.MULTILINE)
    class_metrics = {}
    for cls, iou, f1, ppv, sens, aupr in class_matches[-4:]:
        class_metrics[cls] = {
            'IoU': iou,
            'F1': f1,
            'PPV': ppv,
            'S': sens,
            'AUPR': aupr,
        }

    return {
        'metrics': metrics,
        'class_metrics': class_metrics,
        'log_path': str(log_path),
        'duration_seconds': parse_duration(text),
    }


def result_record(method_label: str, run_name: str, summary: dict) -> dict:
    metrics = summary.get('metrics', {})
    class_metrics = summary.get('class_metrics', {})
    return {
        'run_name': run_name,
        'method': method_label,
        'mIoU': metrics.get('mIoU', ''),
        'mAUPR': metrics.get('mAUPR', ''),
        'mF1': metrics.get('mF1', ''),
        'duration': format_duration(summary.get('duration_seconds')),
        'EX_IoU': class_metrics.get('EX', {}).get('IoU', ''),
        'HE_IoU': class_metrics.get('HE', {}).get('IoU', ''),
        'SE_IoU': class_metrics.get('SE', {}).get('IoU', ''),
        'MA_IoU': class_metrics.get('MA', {}).get('IoU', ''),
        'log_path': summary.get('log_path', ''),
    }


def load_existing_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)


def save_results(records: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['run_name', 'method', 'mIoU', 'mAUPR', 'mF1', 'duration',
                  'EX_IoU', 'HE_IoU', 'SE_IoU', 'MA_IoU', 'log_path']
    with RESULTS_CSV.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        '# LA-M2MRF IDRiD Comparison Table',
        '',
        '| Method | mIoU | mAUPR | mF1 | Time | EX IoU | HE IoU | SE IoU | MA IoU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in records:
        lines.append(
            f'| {row["method"]} | {row["mIoU"]} | {row["mAUPR"]} | {row["mF1"]} | '
            f'{row["duration"]} | {row["EX_IoU"]} | {row["HE_IoU"]} | {row["SE_IoU"]} | {row["MA_IoU"]} |')
    lines.extend([
        '',
        'Logs:',
    ])
    for row in records:
        lines.append(f'- {row["method"]}: {row["log_path"]}')
    RESULTS_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def update_results_table(method_label: str, run_name: str, summary: dict) -> dict:
    records = load_existing_results()
    new_record = result_record(method_label, run_name, summary)
    records = [row for row in records if row.get('run_name') != run_name]
    records.append(new_record)
    order = {
        'idrid_baseline_copy': 0,
        'idrid_full': 1,
        'idrid_la_sampler': 2,
        'idrid_weighted_dice': 3,
        'idrid_fundus_enhance': 4,
    }
    records.sort(key=lambda row: order.get(row.get('run_name', ''), 999))
    save_results(records)
    return new_record


def format_discord_message(subject: str, run_name: str, method_label: str, summary: dict, next_action: str) -> str:
    metrics = summary.get('metrics', {})
    class_metrics = summary.get('class_metrics', {})
    duration = format_duration(summary.get('duration_seconds'))

    metric_line = '、'.join(
        f'{key}={value}' for key, value in metrics.items()) if metrics else '尚未從 log 擷取到 summary metrics'
    class_lines = []
    for cls in ['EX', 'HE', 'SE', 'MA']:
        if cls in class_metrics:
            cls_metrics = class_metrics[cls]
            class_lines.append(
                f'- {cls}: IoU={cls_metrics["IoU"]}, F1={cls_metrics["F1"]}, AUPR={cls_metrics["AUPR"]}')

    body = [
        'Repo：M2MRF-Lesion-Segmentation',
        f'主旨：{subject}',
        '',
        f'目前任務：{method_label} 已完成並產出可比較的 IDRiD 結果。',
        '',
        '我看到的狀態：',
        f'- 整體指標：{metric_line}',
        f'- 總耗時：{duration}',
    ]
    body.extend(class_lines if class_lines else ['- 類別指標：尚未從 log 擷取到'])
    body.append(f'- log：{summary.get("log_path", "unknown")}')
    body.append(f'- 累積表格：{RESULTS_MD}')
    body.append('')
    body.append('下一步：')
    body.append(next_action)
    return '\n'.join(body)


def send_discord(message: str, route: str, attachment: Optional[Path] = None) -> None:
    cmd = [sys.executable, str(DISCORD_SENDER), '--route', route, '--content', message]
    if attachment and attachment.exists():
        cmd.extend(['--file', str(attachment)])
    subprocess.run(cmd, cwd=DISCORD_ROOT, check=True)


def summarize_and_notify(subject: str, run_name: str, method_label: str,
                         work_dir: Path, route: str, next_action: str) -> None:
    log_path = latest_log_file(work_dir)
    if log_path is None:
        message = '\n'.join([
            'Repo：M2MRF-Lesion-Segmentation',
            f'主旨：{subject}',
            '',
            f'目前任務：{run_name} 已結束，但目前找不到對應 log。',
            '',
            '我看到的狀態：',
            f'- work_dir：{work_dir}',
            '',
            '下一步：',
            next_action,
        ])
        send_discord(message, route, RESULTS_MD)
        return

    summary = parse_summary(log_path)
    update_results_table(method_label, run_name, summary)
    message = format_discord_message(subject, run_name, method_label, summary, next_action)
    send_discord(message, route, RESULTS_MD)


def wait_for_pid(pid: int, interval: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(interval)


def run_experiment(exp: Experiment, python_bin: str, seed: str, workers_per_gpu: str) -> int:
    work_dir = REPO_ROOT / exp.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'CUDA_VISIBLE_DEVICES': '0',
    })
    cmd = [
        python_bin,
        'tools/train.py',
        exp.config,
        '--work-dir',
        exp.work_dir,
        '--seed',
        seed,
        '--deterministic',
        '--options',
        f'data.workers_per_gpu={workers_per_gpu}',
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
    return proc.wait()


def main():
    args = parse_args()

    if args.wait_pid:
        wait_for_pid(args.wait_pid, args.sleep_seconds)
        summarize_and_notify(
            subject='Baseline 完成',
            run_name=args.wait_name,
            method_label='M2MRF-C baseline',
            work_dir=REPO_ROOT / args.wait_work_dir,
            route=args.route,
            next_action='我會自動開始跑 LA-M2MRF full，接著依序跑三個 ablation。',
        )

    for index, exp in enumerate(EXPERIMENTS):
        return_code = run_experiment(exp, args.python_bin, args.seed, args.workers_per_gpu)
        if return_code != 0:
            failure_message = '\n'.join([
                'Repo：M2MRF-Lesion-Segmentation',
                f'主旨：{exp.subject} 失敗',
                '',
                f'目前任務：{exp.name} 執行失敗。',
                '',
                '我看到的狀態：',
                f'- return code：{return_code}',
                f'- work_dir：{REPO_ROOT / exp.work_dir}',
                '',
                '下一步：',
                '請回來看該實驗的 log；queue 已停止，避免把後續結果混在錯誤狀態上。',
            ])
            send_discord(failure_message, args.route)
            return return_code

        if index + 1 < len(EXPERIMENTS):
            next_action = f'我會自動接著跑下一組：{EXPERIMENTS[index + 1].name}。'
        else:
            next_action = '五組 IDRiD 實驗已跑完，可以開始整理論文主表與 ablation 表。'
        summarize_and_notify(
            subject=exp.subject,
            run_name=exp.name,
            method_label=exp.method_label,
            work_dir=REPO_ROOT / exp.work_dir,
            route=args.route,
            next_action=next_action,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
