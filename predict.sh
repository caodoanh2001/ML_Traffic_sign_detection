#!/bin/bash
echo "Build source detectron2"
python -m pip install -e .
echo "Prediction"
python detect.py
echo "Export file submission.json. Dir: result/submission.json"
python submission.py