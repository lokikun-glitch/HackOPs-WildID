# 🐅 Segment_Classify: Tiger Individuality Identification & Texture Accumulation Suite

A self-contained, lightweight, peer-shareable toolkit for automated tiger segmentation, individual re-identification, and incremental multi-angle texture template accumulation.

---

## 📦 Directory Structure

```
Segment_Classify/
├── models/
│   ├── tiger_segmenter_best.pt        # Trained YOLOv8n-seg model (6.7 MB)
│   └── tiger_densepose_uv_best.pt     # Trained DensePose UV coordinate model (29.7 MB)
├── input_images/                      # Drop your camera/render photos here
├── gallery/                           # Cumulative individual profiles & merged templates
│   ├── tiger_gallery.json             # Dynamic gallery identity database
│   ├── Tiger_01_template_stripes.png  # Accumulated individual stripe template
│   └── Tiger_01_seen_mask.png         # Observation visibility mask
├── output_templates/                  # Per-image unpacked templates
│   ├── <image>_observed_skin.png      # Exact visible fur on canonical template
│   └── <image>_observed_stripes.png   # Exact isolated melanin stripe markings
├── tiger_extractor.py                 # Segmentation & stripe marking extraction
├── tiger_unpacker.py                  # Canonical UV template unrolling (preserves gaps)
├── tiger_gallery.py                   # Gallery matching & threshold-gated stitching
├── run_pipeline.py                    # Main batch execution script
├── tiger_segmentation_demo.ipynb      # Complete 22-cell interactive demo notebook
├── run.bat                            # 1-Click execution script for Windows
└── README.md                          # Complete documentation and quickstart
```

---

## 🚀 Quickstart

### 1. Requirements
Ensure your Python environment has the following installed:
```bash
pip install ultralytics opencv-python numpy torch torchvision matplotlib
```

### 2. Run with 1-Click (Windows)
Double-click `run.bat` or run from PowerShell / Terminal:
```bash
python run_pipeline.py
```

### 3. Add New Images Over Time
Simply drop any new camera photos into `input_images/` and re-run:
```bash
python run_pipeline.py --input input_images/
```

- If an image matches an existing tiger ($\ge 0.70$ similarity), its newly observed surface is automatically stitched into that tiger's cumulative template in `gallery/`.
- If an image belongs to an unseen tiger ($< 0.70$), a new profile (`Tiger_02`, `Tiger_03`, ...) is automatically created.

---

## 🔑 Core Features & Principles

1. **Strict Incompleteness (No Hallucination)**:
   - If any body part is occluded by trees, foliage, or out of frame, **that area remains 100% blank white (`[255, 255, 255]`)**.
   - Zero synthetic patterns or default skins are ever hallucinated.
2. **Threshold-Gated Stitching**:
   - Multiple camera passes only merge into a single individual template when their overlapping stripe texture intersection similarity exceeds the trained threshold ($\ge 0.70$).
3. **Completely Self-Contained**:
   - Contains all trained weights (`models/`), inference scripts, sample images, and the complete interactive Jupyter notebook (`tiger_segmentation_demo.ipynb`).
