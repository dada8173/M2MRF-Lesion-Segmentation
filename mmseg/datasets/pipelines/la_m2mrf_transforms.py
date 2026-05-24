import cv2
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LesionAwareRandomCrop(object):
    """Random crop with optional lesion-prioritized sampling."""

    def __init__(self,
                 crop_size,
                 lesion_prob=0.7,
                 small_lesion_prob=0.4,
                 target_classes=(1, 2, 3, 4),
                 priority_classes=(3, 4),
                 cat_max_ratio=1.,
                 ignore_index=255,
                 num_retry=10):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.lesion_prob = lesion_prob
        self.small_lesion_prob = small_lesion_prob
        self.target_classes = tuple(target_classes)
        self.priority_classes = tuple(priority_classes)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.num_retry = num_retry

    def crop(self, img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        return img[crop_y1:crop_y2, crop_x1:crop_x2, ...]

    def get_random_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return (offset_h, offset_h + self.crop_size[0],
                offset_w, offset_w + self.crop_size[1])

    def _sample_axis_start(self, coord, crop_len, img_len):
        if img_len <= crop_len:
            return 0
        min_start = max(coord - crop_len + 1, 0)
        max_start = min(coord, img_len - crop_len)
        if max_start < min_start:
            return max(min(coord, img_len - crop_len), 0)
        return np.random.randint(min_start, max_start + 1)

    def get_lesion_crop_bbox(self, seg_map, candidate_classes):
        if len(candidate_classes) == 0:
            return None
        lesion_mask = np.isin(seg_map, np.array(candidate_classes))
        lesion_coords = np.argwhere(lesion_mask)
        if lesion_coords.size == 0:
            return None

        coord_idx = np.random.randint(0, lesion_coords.shape[0])
        center_y, center_x = lesion_coords[coord_idx]
        crop_y1 = self._sample_axis_start(center_y, self.crop_size[0], seg_map.shape[0])
        crop_x1 = self._sample_axis_start(center_x, self.crop_size[1], seg_map.shape[1])
        return (crop_y1, crop_y1 + self.crop_size[0],
                crop_x1, crop_x1 + self.crop_size[1])

    def _pick_candidate_classes(self, seg_map):
        if np.random.rand() < self.small_lesion_prob:
            priority_present = [cls for cls in self.priority_classes if np.any(seg_map == cls)]
            if len(priority_present) > 0:
                return priority_present
        return [cls for cls in self.target_classes if np.any(seg_map == cls)]

    def _cat_ratio_ok(self, seg_map, crop_bbox):
        if seg_map is None:
            return True
        if self.cat_max_ratio >= 1.:
            return True
        seg_temp = self.crop(seg_map, crop_bbox)
        labels, cnt = np.unique(seg_temp, return_counts=True)
        cnt = cnt[labels != self.ignore_index]
        return len(cnt) <= 1 or np.max(cnt) / np.sum(cnt) < self.cat_max_ratio

    def __call__(self, results):
        img = results['img']
        seg_map = results.get('gt_semantic_seg', None)

        crop_bbox = self.get_random_crop_bbox(img)
        use_lesion_sampling = (
            seg_map is not None and np.random.rand() < self.lesion_prob)

        if use_lesion_sampling:
            candidate_classes = self._pick_candidate_classes(seg_map)
            for _ in range(self.num_retry):
                lesion_bbox = self.get_lesion_crop_bbox(seg_map, candidate_classes)
                if lesion_bbox is None:
                    break
                if self._cat_ratio_ok(seg_map, lesion_bbox):
                    crop_bbox = lesion_bbox
                    break
            else:
                candidate_classes = []

        if not self._cat_ratio_ok(seg_map, crop_bbox):
            for _ in range(self.num_retry):
                random_bbox = self.get_random_crop_bbox(img)
                if self._cat_ratio_ok(seg_map, random_bbox):
                    crop_bbox = random_bbox
                    break

        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape

        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                f'lesion_prob={self.lesion_prob}, '
                f'small_lesion_prob={self.small_lesion_prob}, '
                f'target_classes={self.target_classes}, '
                f'priority_classes={self.priority_classes}, '
                f'cat_max_ratio={self.cat_max_ratio}, '
                f'num_retry={self.num_retry})')


@PIPELINES.register_module()
class FundusCLAHEEnhancement(object):
    """Apply CLAHE to the green channel and keep 3-channel output."""

    def __init__(self,
                 apply_prob=1.0,
                 mode='green',
                 clip_limit=2.0,
                 tile_grid_size=(8, 8)):
        assert 0.0 <= apply_prob <= 1.0
        if mode != 'green':
            raise ValueError('Only green mode is supported in the first LA-M2MRF version.')
        self.apply_prob = apply_prob
        self.mode = mode
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)

    def _to_uint8(self, img):
        orig_dtype = img.dtype
        if np.issubdtype(orig_dtype, np.floating):
            img_float = img.astype(np.float32)
            if img_float.max() <= 1.0 and img_float.min() >= 0.0:
                img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
                return img_uint8, orig_dtype, True
            img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
            return img_uint8, orig_dtype, False
        return np.clip(img, 0, 255).astype(np.uint8), orig_dtype, False

    def _restore_dtype(self, img_uint8, orig_dtype, was_unit_float):
        if np.issubdtype(orig_dtype, np.floating):
            img = img_uint8.astype(np.float32)
            if was_unit_float:
                img = img / 255.0
            return img.astype(orig_dtype, copy=False)
        return img_uint8.astype(orig_dtype, copy=False)

    def __call__(self, results):
        if np.random.rand() >= self.apply_prob:
            return results

        img = results['img']
        if img.ndim != 3 or img.shape[2] != 3:
            return results

        img_uint8, orig_dtype, was_unit_float = self._to_uint8(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size)
        enhanced = img_uint8.copy()
        enhanced[:, :, 1] = clahe.apply(img_uint8[:, :, 1])
        results['img'] = self._restore_dtype(enhanced, orig_dtype, was_unit_float)
        results['img_shape'] = results['img'].shape
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(apply_prob={self.apply_prob}, '
                f"mode='{self.mode}', clip_limit={self.clip_limit}, "
                f'tile_grid_size={self.tile_grid_size})')
