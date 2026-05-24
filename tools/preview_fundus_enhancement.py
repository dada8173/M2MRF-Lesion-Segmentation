import argparse
import os

import cv2
import numpy as np


def apply_green_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Mirror the repo's green-channel CLAHE enhancement."""
    if img_bgr is None:
        raise ValueError('Input image is empty.')
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError('Expected a 3-channel BGR image.')

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid_size))
    enhanced = img_bgr.copy()
    enhanced[:, :, 1] = clahe.apply(img_bgr[:, :, 1])
    return enhanced


def build_comparison_panel(original, enhanced, gap=24):
    height = max(original.shape[0], enhanced.shape[0]) + 80
    width = original.shape[1] + enhanced.shape[1] + gap * 3
    panel = np.full((height, width, 3), 255, dtype=np.uint8)

    x1 = gap
    x2 = x1 + original.shape[1] + gap
    y = 56

    panel[y:y + original.shape[0], x1:x1 + original.shape[1]] = original
    panel[y:y + enhanced.shape[0], x2:x2 + enhanced.shape[1]] = enhanced

    cv2.putText(panel, 'Original', (x1, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Green-channel CLAHE', (x2, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preview fundus lesion enhancement with green-channel CLAHE.')
    parser.add_argument(
        '--input',
        default='../data/IDRID/image/train/IDRiD_36.jpg',
        help='Input fundus image path.')
    parser.add_argument(
        '--output-dir',
        default='work_dirs/la_m2mrf/fundus_enhance_preview',
        help='Directory for preview outputs.')
    parser.add_argument(
        '--clip-limit',
        type=float,
        default=2.0,
        help='CLAHE clip limit.')
    parser.add_argument(
        '--tile-grid-size',
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=('H', 'W'),
        help='CLAHE tile grid size.')
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError('Input image not found: {}'.format(args.input))

    os.makedirs(args.output_dir, exist_ok=True)

    original = cv2.imread(args.input, cv2.IMREAD_COLOR)
    enhanced = apply_green_clahe(
        original,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid_size)
    panel = build_comparison_panel(original, enhanced)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    original_path = os.path.join(args.output_dir, stem + '_original.jpg')
    enhanced_path = os.path.join(args.output_dir, stem + '_enhanced.jpg')
    panel_path = os.path.join(args.output_dir, stem + '_comparison.jpg')

    cv2.imwrite(original_path, original)
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(panel_path, panel)

    print('Saved original:', original_path)
    print('Saved enhanced:', enhanced_path)
    print('Saved comparison:', panel_path)


if __name__ == '__main__':
    main()
