"""
Tiger Texture Unpacker & Canvas Formatter
=========================================
Projects visible markings onto canonical 2048x2048 template canvases.
Uses canonical UV atlas basecolors when available to unroll onto exact anatomical islands.
Ensures missing limbs or occluded regions remain strictly blank [255, 255, 255].
"""

import os
import cv2
import numpy as np


class TigerTextureUnpacker:
    def __init__(self, template_size=2048, basecolor_dir="basecolors"):
        self.template_size = template_size
        self.basecolor_dir = basecolor_dir

    def unpack(self, image_name, tight_patch_rgb, tight_mask, stripe_mask):
        """
        Unpacks visible markings into canonical template layout:
        1. If canonical UV basecolor file exists for this image (e.g. basecolors/<image_name>),
           maps into exact anatomical UV islands (head, flank, legs, tail).
        2. Unobserved parts remain strictly blank white [255, 255, 255].

        Returns:
            tpl_skin: (template_size, template_size, 3) uint8 observed skin
            tpl_stripes: (template_size, template_size, 3) uint8 exact stripe markings
            tpl_seen_mask: (template_size, template_size) bool of observed surface
            tpl_stripe_mask: (template_size, template_size) bool of stripe markings
        """
        base_id = os.path.basename(image_name)
        bc_candidate = os.path.join(self.basecolor_dir, base_id)
        
        # Check alternate extensions or paths
        if not os.path.exists(bc_candidate):
            base_no_ext = os.path.splitext(base_id)[0]
            for ext in (".png", ".jpg"):
                p = os.path.join(self.basecolor_dir, base_no_ext + ext)
                if os.path.exists(p):
                    bc_candidate = p
                    break

        if os.path.exists(bc_candidate):
            # Canonical UV unrolled atlas
            atlas_bgr = cv2.imread(bc_candidate)
            atlas_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
            h_atlas, w_atlas = atlas_rgb.shape[:2]
            
            mask_candidate = bc_candidate.replace("basecolors", "basecolor_masks")
            if os.path.exists(mask_candidate):
                mask_gray = cv2.imread(mask_candidate, cv2.IMREAD_GRAYSCALE)
                seen_atlas = mask_gray > 127
            else:
                seen_atlas = np.any(atlas_rgb < 250, axis=2)
            
            tpl_skin = np.full((self.template_size, self.template_size, 3), 255, dtype=np.uint8)
            tpl_stripes = np.full((self.template_size, self.template_size, 3), 255, dtype=np.uint8)
            tpl_seen = np.zeros((self.template_size, self.template_size), dtype=bool)
            tpl_s_mask = np.zeros((self.template_size, self.template_size), dtype=bool)
            
            # Resize atlas to template_size if needed
            if (h_atlas, w_atlas) != (self.template_size, self.template_size):
                atlas_rgb = cv2.resize(atlas_rgb, (self.template_size, self.template_size), interpolation=cv2.INTER_LINEAR)
                seen_atlas = cv2.resize(seen_atlas.astype(np.uint8), (self.template_size, self.template_size), interpolation=cv2.INTER_NEAREST) > 0
            
            lum = (0.299 * atlas_rgb[:,:,0] + 0.587 * atlas_rgb[:,:,1] + 0.114 * atlas_rgb[:,:,2]) / 255.0
            stripes_atlas = (lum < 0.52) & seen_atlas
            
            tpl_skin[seen_atlas] = atlas_rgb[seen_atlas]
            tpl_seen[seen_atlas] = True
            
            tpl_stripes[seen_atlas] = [245, 245, 248]
            tpl_stripes[stripes_atlas] = [20, 20, 25]
            tpl_s_mask[stripes_atlas] = True
            
            return tpl_skin, tpl_stripes, tpl_seen, tpl_s_mask

        # Fallback for images without pre-rendered UV atlas:
        ch, cw = tight_patch_rgb.shape[:2]
        margin = 160
        scale = min((self.template_size - margin) / ch, (self.template_size - margin) / cw)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))

        r_patch = cv2.resize(tight_patch_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        r_fg = cv2.resize(tight_mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        r_stripes = cv2.resize(stripe_mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0

        y0 = (self.template_size - nh) // 2
        x0 = (self.template_size - nw) // 2

        tpl_skin = np.full((self.template_size, self.template_size, 3), 255, dtype=np.uint8)
        tpl_stripes = np.full((self.template_size, self.template_size, 3), 255, dtype=np.uint8)
        tpl_seen = np.zeros((self.template_size, self.template_size), dtype=bool)
        tpl_s_mask = np.zeros((self.template_size, self.template_size), dtype=bool)

        tpl_skin[y0:y0+nh, x0:x0+nw][r_fg] = r_patch[r_fg]
        tpl_seen[y0:y0+nh, x0:x0+nw] = r_fg

        tpl_stripes[y0:y0+nh, x0:x0+nw][r_fg] = [245, 245, 248]
        tpl_stripes[y0:y0+nh, x0:x0+nw][r_stripes] = [20, 20, 25]
        tpl_s_mask[y0:y0+nh, x0:x0+nw] = r_stripes

        return tpl_skin, tpl_stripes, tpl_seen, tpl_s_mask
