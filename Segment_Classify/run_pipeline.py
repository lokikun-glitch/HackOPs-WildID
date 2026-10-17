"""
Automated Incremental Tiger Identification Pipeline
===================================================
Processes single images or entire folders of tiger photos:
1. Detects and segments foreground tigers (ignoring complex forest backgrounds).
2. Extracts dark melanin stripe patterns.
3. Unrolls visible markings into canonical 2048x2048 UV template layout.
4. Matches against the cumulative individual gallery using UV stripe intersection:
   - If similarity >= threshold (default 0.85) -> Stitches into existing tiger profile.
   - If similarity <  threshold -> Creates new tiger profile.
"""

import os
import sys
import glob
import argparse
import cv2

from tiger_extractor import TigerSegmenterExtractor
from tiger_unpacker import TigerTextureUnpacker
from tiger_gallery import TigerGalleryManager


def process_pipeline(
    input_source="input_images",
    model_path="models/tiger_segmenter_best.pt",
    output_dir="output_templates",
    gallery_dir="gallery",
    basecolor_dir="basecolors",
    similarity_thresh=0.85
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    print("=" * 70)
    print("🐅 TIGER INDIVIDUALITY IDENTIFIER & ACCUMULATION PIPELINE")
    print("=" * 70)

    # Initialize modules
    print("[1/3] Loading trained segmenter model...")
    segmenter = TigerSegmenterExtractor(model_path=model_path)

    print("[2/3] Initializing canonical texture unpacker...")
    unpacker = TigerTextureUnpacker(template_size=2048, basecolor_dir=basecolor_dir)

    print(f"[3/3] Connecting to gallery manager (Trained Threshold = {similarity_thresh:.2f})...")
    gallery = TigerGalleryManager(gallery_dir=gallery_dir, similarity_threshold=similarity_thresh)

    # Collect images
    if os.path.isdir(input_source):
        img_paths = sorted(
            glob.glob(os.path.join(input_source, "*.png")) +
            glob.glob(os.path.join(input_source, "*.jpg")) +
            glob.glob(os.path.join(input_source, "*.jpeg"))
        )
    elif os.path.isfile(input_source):
        img_paths = [input_source]
    else:
        print(f"[ERROR] Input source not found: {input_source}")
        return

    print(f"\nProcessing {len(img_paths)} incoming images...\n")

    for i, path in enumerate(img_paths, 1):
        name = os.path.basename(path)
        print(f"[{i}/{len(img_paths)}] Processing: {name}")

        # 1. Segment & extract stripes
        res = segmenter.process_image(path)
        if res['tiger_pixels'] == 0:
            print(f"  [WARN] No tiger detected in {name}. Skipping.")
            continue

        print(f"  -> Detected Tiger: Conf={res['confidence']:.1%}, Surface={res['tiger_pixels']:,} px, Stripes={res['stripe_count']:,} px")

        # 2. Unpack to canonical template canvas
        tpl_skin, tpl_stripes, tpl_seen, tpl_s_mask = unpacker.unpack(
            name, res['tight_patch'], res['tight_mask'], res['stripe_mask']
        )

        # Save single-pass unpacked observation
        base_id = os.path.splitext(name)[0]
        obs_skin_file = os.path.join(output_dir, f"{base_id}_observed_skin.png")
        obs_stripes_file = os.path.join(output_dir, f"{base_id}_observed_stripes.png")
        cv2.imwrite(obs_skin_file, cv2.cvtColor(tpl_skin, cv2.COLOR_RGB2BGR))
        cv2.imwrite(obs_stripes_file, cv2.cvtColor(tpl_stripes, cv2.COLOR_RGB2BGR))

        # 3. Match against gallery & update cumulative individual profile
        match_info = gallery.match_or_register(
            name,
            tpl_skin,
            tpl_stripes,
            tpl_seen,
            tpl_s_mask
        )

        print(f"  -> Outcome: {match_info['decision']} | Assigned to: {match_info['individual_id']} (Total Gallery: {match_info['total_gallery_tigers']} tigers)\n")

    print("=" * 70)
    print("✅ PIPELINE EXECUTION COMPLETED!")
    print(f"  Gallery Database: {os.path.join(gallery_dir, 'tiger_gallery.json')}")
    print(f"  Output Templates: {output_dir}/")
    print(f"  Total Recognized Individuals: {len(gallery.gallery)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiger Individuality Identifier & Template Stitcher")
    parser.add_argument("--input", default="input_images", help="Path to input image or directory")
    parser.add_argument("--models", default="models/tiger_segmenter_best.pt", help="Path to YOLO segmenter weights")
    parser.add_argument("--out", default="output_templates", help="Output directory for single templates")
    parser.add_argument("--gallery", default="gallery", help="Directory for cumulative individual profiles")
    parser.add_argument("--basecolors", default="basecolors", help="Directory containing canonical basecolors")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold for matching")

    args = parser.parse_args()
    process_pipeline(
        input_source=args.input,
        model_path=args.models,
        output_dir=args.out,
        gallery_dir=args.gallery,
        basecolor_dir=args.basecolors,
        similarity_thresh=args.threshold
    )
