# 🐅 WildID: AI-Powered Tiger Individuality Identification & Territory Intelligence Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-Runtime-lightgrey.svg)](https://palletsprojects.com/p/flask/)
[![React](https://img.shields.io/badge/React-18%2B-blue.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-Frontend-purple.svg)](https://vitejs.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-yellow.svg)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)

An end-to-end evidence-based wildlife analytics and conservation platform combining:
1. **Computer Vision & Texture Accumulation**: Automated tiger segmentation, DensePose canonical UV coordinate projection, individual stripe pattern extraction, and multi-angle incremental template accumulation.
2. **Territory & Behaviour Intelligence**: Camera-trap sighting synthesis, home range KDE estimation, overlap analysis, conflict risk prediction, and annual intelligence reporting.

---

## 🏗️ Repository Architecture

```
HackOPs-WildID/
├── Merge/
│   ├── FINAL.bat                  # 1-Click launcher for Flask-based Tiger Intelligence System
│   ├── run_flask.py               # Unified Flask server runner (Zero Node.js dependency)
│   ├── Behaviour_Model/           # Territory and behavioural models & interactive visualizer
│   │   ├── generate_tiger_data.py # Synthetic sighting & trajectory generation
│   │   ├── tiger_planner.py       # Behavioural trajectory planning & risk analysis
│   │   ├── tadoba_territory_map.html # Interactive GIS territory mapping
│   │   └── output/                # Pre-computed sightings, predictions, and geojson
│   └── VNAVTASK/                  # Full-stack Tiger Territory Intelligence System
│       ├── backend/               # FastAPI backend with SQLite, analytics & report generators
│       ├── flask_app/             # Embedded Flask web server & intelligence API
│       ├── frontend/              # Modern React + Vite + TypeScript dashboard
│       ├── data/                  # Tiger profiles, gallery templates, synthetic benchmarks
│       └── models/                # Territory estimation & tracking models
│
├── Segment_Classify/              # Computer Vision & Re-Identification Suite
│   ├── models/                    # YOLOv8-seg and DensePose UV deep learning weights
│   ├── input_images/              # Camera trap and observation image inputs
│   ├── gallery/                   # Accumulated identity profiles & canonical stripe templates
│   ├── output_templates/          # Observed skin and isolated melanin markings
│   ├── tiger_extractor.py         # Segmentation & melanin stripe isolation
│   ├── tiger_unpacker.py          # Canonical 2048x2048 UV template projection
│   ├── tiger_gallery.py           # Similarity matching & threshold-gated stitching
│   ├── run_pipeline.py            # End-to-end batch processing pipeline
│   ├── tiger_segmentation_demo.ipynb # 22-cell interactive demo notebook
│   └── run.bat                    # 1-Click execution script for Windows
│
└── dataset_segmentation_pairs/    # Dataset & Annotation Artifacts
    ├── dataset_joints.sqlite      # SQLite database of annotated joints and keypoints
    ├── dataset_metadata.csv       # Sighting and image metadata
    ├── images/                    # Ground-truth tiger images
    ├── masks/                     # Binary segmentation masks
    └── previews/                  # Visual verification previews
```

---

## 🚀 Quickstart Guides

### Option 1: Tiger Territory Intelligence System (1-Click Launch)
Double-click **`Merge\FINAL.bat`** (Windows) or run from terminal:
```bash
python Merge/run_flask.py
```
- Open browser at `http://127.0.0.1:5000`
- Provides interactive intelligence dashboards, territory maps, tiger profiles, and automated PDF/CSV report generation.

### Option 2: Tiger Individuality Identification Pipeline
Run automated segmentation, UV unrolling, and gallery accumulation on input camera photos:
```bash
cd Segment_Classify
python run_pipeline.py --input input_images/
```
- Matched tigers ($\ge 0.70$ similarity) incrementally stitch observed markings into their cumulative identity template in `gallery/`.
- Unseen tigers ($< 0.70$) automatically initialize new identity templates (`Tiger_01`, `Tiger_02`, ...).

### Option 3: Full-Stack React + FastAPI Dashboard

**Backend (FastAPI):**
```bash
cd Merge/VNAVTASK
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```
Interactive API Swagger Docs: `http://127.0.0.1:8000/docs`

**Frontend (React + Vite + TypeScript):**
```bash
cd Merge/VNAVTASK/frontend
pnpm install
pnpm run dev
```
Frontend runs at `http://localhost:5173`

---

## 🔬 Core Methodologies

1. **Strict Incompleteness & Zero Hallucination**:
   Unobserved tiger body sections remain strictly unobserved without synthetic hallucination, ensuring reliable forensic identification.
2. **Canonical DensePose UV Unrolling**:
   Flank, leg, head, and tail stripes are normalized into anatomical UV atlas coordinates, enabling rotation- and pose-invariant matching.
3. **Evidence-Based Spatial Intelligence**:
   Kernel Density Estimation (KDE) and trajectory planning track home ranges, encounter risks, and dispersal patterns across protected wildlife zones (e.g., Tadoba-Andhari Tiger Reserve).

---

## 📄 License
Developed for HackOPs. For conservation and academic research use.
