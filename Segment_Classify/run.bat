@echo off
title Tiger Segment & Classify Pipeline
echo ======================================================================
echo 🐅 TIGER INDIVIDUALITY IDENTIFIER ^& ACCUMULATION PIPELINE
echo ======================================================================
python run_pipeline.py --input input_images --models models/tiger_segmenter_best.pt --out output_templates --gallery gallery
pause
