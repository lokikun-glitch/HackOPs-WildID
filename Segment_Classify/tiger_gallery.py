"""
Individual Tiger Gallery & Re-Identification Matcher
===================================================
Maintains a dynamic gallery of individual tigers.
- Uses Canonical UV Texture Intersection for high-precision individual matching.
- Only when overlapping surface stripe similarity >= threshold (default 0.85):
    -> Confirmed MATCH to that individual tiger.
    -> Newly observed surface is stitched into the individual template.
- If similarity < threshold:
    -> Discovered as a NEW individual tiger with its own incomplete template.
"""

import os
import json
import cv2
import numpy as np


class TigerGalleryManager:
    def __init__(
        self,
        gallery_dir="gallery",
        similarity_threshold=0.85
    ):
        self.gallery_dir = gallery_dir
        self.threshold = similarity_threshold
        self.db_file = os.path.join(gallery_dir, "tiger_gallery.json")
        os.makedirs(gallery_dir, exist_ok=True)
        self.gallery = self._load_gallery()

    def _load_gallery(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_gallery(self):
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.gallery, f, indent=2)

    def compute_uv_stripe_intersection(self, s1, seen1, s2, seen2):
        """
        Computes stripe intersection over union on the mutually observed surface.
        """
        common = np.logical_and(seen1, seen2)
        common_px = int(common.sum())
        if common_px < 2000:
            return 0.0, common_px

        s_inter = np.logical_and(s1, s2).sum()
        s_union = np.logical_or(s1 & common, s2 & common).sum()
        sim = float(s_inter) / max(1.0, float(s_union))
        return sim, common_px

    def match_or_register(
        self,
        image_name,
        tpl_skin,
        tpl_stripes,
        tpl_seen,
        tpl_s_mask
    ):
        """
        Matches incoming observation against existing gallery individuals using UV stripe intersection.
        """
        best_id = None
        best_sim = -1.0
        best_common = 0

        for tiger_id, data in self.gallery.items():
            cum_mask_path = os.path.join(self.gallery_dir, f"{tiger_id}_seen_mask.png")
            cum_stripes_path = os.path.join(self.gallery_dir, f"{tiger_id}_template_stripes.png")

            if os.path.exists(cum_mask_path) and os.path.exists(cum_stripes_path):
                ref_mask = cv2.imread(cum_mask_path, cv2.IMREAD_GRAYSCALE) > 127
                ref_stripes_img = cv2.imread(cum_stripes_path)
                ref_lum = (0.299 * ref_stripes_img[:,:,0] + 0.587 * ref_stripes_img[:,:,1] + 0.114 * ref_stripes_img[:,:,2]) / 255.0
                ref_stripes = (ref_lum < 0.30) & ref_mask

                sim, common_px = self.compute_uv_stripe_intersection(
                    tpl_s_mask, tpl_seen, ref_stripes, ref_mask
                )

                if sim > best_sim:
                    best_sim = sim
                    best_id = tiger_id
                    best_common = common_px

        is_match = (best_sim >= self.threshold) and (best_id is not None)

        if is_match:
            assigned_id = best_id
            decision = "MATCH"
            print(f"  [MATCH] '{image_name}' matched to existing individual: {assigned_id} (Stripe IoU: {best_sim:.4f} >= {self.threshold:.2f}, Overlap: {best_common:,} px)")

            tiger_entry = self.gallery[assigned_id]
            cum_stripes_path = os.path.join(self.gallery_dir, f"{assigned_id}_template_stripes.png")
            cum_skin_path = os.path.join(self.gallery_dir, f"{assigned_id}_template_skin.png")
            cum_mask_path = os.path.join(self.gallery_dir, f"{assigned_id}_seen_mask.png")

            cum_stripes = cv2.imread(cum_stripes_path)
            cum_skin = cv2.imread(cum_skin_path) if os.path.exists(cum_skin_path) else np.full_like(cum_stripes, 255)
            cum_seen = cv2.imread(cum_mask_path, cv2.IMREAD_GRAYSCALE) > 127

            # Merge newly observed surface into cumulative template
            newly_seen = tpl_seen & (~cum_seen)
            cum_skin[newly_seen] = cv2.cvtColor(tpl_skin, cv2.COLOR_RGB2BGR)[newly_seen]
            cum_stripes[newly_seen] = [245, 245, 248]
            cum_stripes[tpl_s_mask] = [20, 20, 25]
            cum_seen |= tpl_seen

            cv2.imwrite(cum_stripes_path, cum_stripes)
            cv2.imwrite(cum_skin_path, cum_skin)
            cv2.imwrite(cum_mask_path, (cum_seen.astype(np.uint8) * 255))

            tiger_entry["observations"].append(image_name)
            tiger_entry["total_texels"] = int(np.sum(cum_seen))

        else:
            # Register NEW Individual
            assigned_id = f"Tiger_{len(self.gallery) + 1:02d}"
            decision = "NEW"
            print(f"  [NEW INDIVIDUAL] '{image_name}' registered as: {assigned_id} (Highest similarity: {best_sim:.4f} < {self.threshold:.2f})")

            cum_stripes_path = os.path.join(self.gallery_dir, f"{assigned_id}_template_stripes.png")
            cum_skin_path = os.path.join(self.gallery_dir, f"{assigned_id}_template_skin.png")
            cum_mask_path = os.path.join(self.gallery_dir, f"{assigned_id}_seen_mask.png")

            cv2.imwrite(cum_stripes_path, cv2.cvtColor(tpl_stripes, cv2.COLOR_RGB2BGR))
            cv2.imwrite(cum_skin_path, cv2.cvtColor(tpl_skin, cv2.COLOR_RGB2BGR))
            cv2.imwrite(cum_mask_path, (tpl_seen.astype(np.uint8) * 255))

            self.gallery[assigned_id] = {
                "individual_id": assigned_id,
                "first_seen": image_name,
                "observations": [image_name],
                "total_texels": int(np.sum(tpl_seen))
            }

        self._save_gallery()

        return {
            "decision": decision,
            "individual_id": assigned_id,
            "confidence": float(best_sim),
            "total_gallery_tigers": len(self.gallery),
            "observations_count": len(self.gallery[assigned_id]["observations"])
        }
