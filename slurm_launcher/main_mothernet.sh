#!/usr/bin/env bash
set -e
# Showing the environment variable that was passed when calling Slurmpilot
source ~/.bashrc
conda activate gamformer
PYTHONPATH=/home/salinasd/ python slurm_launcher/evaluate_openml_datasets.py
