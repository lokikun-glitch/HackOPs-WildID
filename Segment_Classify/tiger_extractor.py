"""
Segmentation & Stripe Marking Extractor Module
===============================================
Extracts tiger silhouette using YOLOv8n-seg, tight foreground crop,
and isolates dark melanin stripe patterns using morphological black-hat filtering.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class TigerSegmenterExtractor:
    """
    Combines deep segmentation with adaptive morphological stripe ridge extraction.
    """

    def __init__(self, model_path="models/tiger_segmenter_best.pt", conf_thresh=0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def process_image(self, image_input):
        """
        Takes an image path or BGR image numpy array.

        Returns:
            dict containing:
                'orig_bgr': (H, W, 3) image
                'orig_rgb': (H, W, 3) image
                'mask_bin': (H, W) uint8 binary mask (255 = tiger)
                'cutout_rgb': (H, W, 3) background zeroed out
                'tight_patch': cropped RGB patch
                'tight_mask': cropped bool mask
                'stripe_mask': cropped bool stripe ridges mask
                'stripe_count': int pixel count of stripes
                'tiger_pixels': int pixel count of tiger
                'bbox': (x1, y1, x2, y2)
                'confidence': float
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            bgr = cv2.imread(image_input)
        else:
            bgr = image_input.copy()

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model.predict(source=bgr, conf=self.conf_thresh, verbose=False)
        res = results[0]

        mask_bin = np.zeros((h, w), dtype=np.uint8)
        bbox = (0, 0, w, h)
        conf = 0.0

        if res.masks is not None and len(res.masks) > 0:
            for poly in res.masks.xy:
                if len(poly) > 0:
                    cv2.fillPoly(mask_bin, [np.int32(poly)], 255)
            if res.boxes is not None and len(res.boxes) > 0:
                bbox = [int(v) for v in res.boxes.xyxy[0]]
                conf = float(res.boxes.conf[0])

        mask_bool = mask_bin > 0
        tiger_pixels = int(np.sum(mask_bool))

        # Cutout
        cutout = np.zeros_like(rgb)
        cutout[mask_bool] = rgb[mask_bool]

        # Tight foreground patch
        if np.any(mask_bool):
            ys, xs = np.nonzero(mask_bool)
            pad = 16
            ymin, ymax = max(0, ys.min() - pad), min(h - 1, ys.max() + pad)
            xmin, xmax = max(0, xs.min() - pad), min(w - 1, xs.max() + pad)
            tight_patch = rgb[ymin:ymax+1, xmin:xmax+1]
            tight_mask = mask_bool[ymin:ymax+1, xmin:xmax+1]
        else:
            tight_patch = rgb
            tight_mask = np.ones((h, w), dtype=bool)

        # Morphological stripe extraction
        gray_patch = cv2.cvtColor(tight_patch, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray_patch)

        k_size = max(5, int(min(tight_patch.shape[:2]) * 0.04))
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        blackhat = cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel)
        blackhat[~tight_mask] = 0

        if np.any(tight_mask):
            thresh_val = np.percentile(blackhat[tight_mask], 65)
            stripe_mask_raw = (blackhat > thresh_val) & tight_mask
        else:
            stripe_mask_raw = np.zeros_like(tight_mask, dtype=bool)

        stripe_mask_uint8 = cv2.morphologyEx(
            stripe_mask_raw.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        stripe_mask = stripe_mask_uint8 > 0
        stripe_count = int(np.sum(stripe_mask))

        return {
            'orig_bgr': bgr,
            'orig_rgb': rgb,
            'mask_bin': mask_bin,
            'mask_bool': mask_bool,
            'cutout_rgb': cutout,
            'tight_patch': tight_patch,
            'tight_mask': tight_mask,
            'stripe_mask': stripe_mask,
            'stripe_count': stripe_count,
            'tiger_pixels': tiger_pixels,
            'bbox': bbox,
            'confidence': conf
        }
