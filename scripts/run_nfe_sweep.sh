#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=2

/data/gjh8760/anaconda3/envs/elir/bin/python eval_nfe_sweep.py \
    -y configs/elir_infer_llie.yaml \
    --k_steps 1 2 4 5 8
