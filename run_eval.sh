#!/usr/bin/env bash
set -e
python -u src/eval.py --config configs/default.yaml --ckpt outputs/checkpoints/best.pt
