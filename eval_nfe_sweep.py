"""
k_steps (NFE) sweep over a trained ELIR checkpoint.

Usage:
    python eval_nfe_sweep.py -y configs/elir_infer_llie.yaml \
        --k_steps 1 2 4 5 8 \
        --out runs/elir_llie_lolv1/nfe_sweep.json
"""
import argparse
import json
import os
import warnings
from copy import deepcopy
from datetime import datetime

import pytorch_lightning as L
from hyperpyyaml import load_hyperpyyaml

from ELIR.datasets.dataset import get_loader
from ELIR.irsetup import IRSetup
from ELIR.models.load_model import get_model
from utils import set_seed

warnings.filterwarnings("ignore")


def override_k_steps(conf, k):
    """Set k_steps in every nested fm_cfg."""
    arch_params = conf["model_cfg"]["arch_cfg"]["params"]
    arch_params["fm_cfg"]["k_steps"] = k
    # top-level fm_cfg may also exist (train config); harmless for eval
    if "fm_cfg" in conf:
        conf["fm_cfg"]["k_steps"] = k


def run_one(conf, k):
    env_cfg = conf["env_cfg"]
    seed = env_cfg.get("seed", 0)
    set_seed(seed)

    valloader = get_loader(conf["dataset_cfg"]["val_dataset"])

    model = get_model(conf["model_cfg"]["arch_cfg"])

    eval_cfg = conf["eval_cfg"]
    eval_cfg["log_images"] = False  # sweep: skip image dump
    eval_cfg.setdefault("max_log_images", 0)

    run_dir = os.path.join("./phase1/nfe_sweep", f"k{k}")
    os.makedirs(run_dir, exist_ok=True)

    setup = IRSetup(model, eval_cfg=eval_cfg, run_dir=run_dir)
    trainer = L.Trainer(logger=False, enable_progress_bar=True)
    set_seed(seed)
    results = trainer.validate(setup, dataloaders=valloader)

    metrics = eval_cfg["metrics"]
    return {m: float(results[0][m]) for m in metrics}, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", "-y", type=str, required=True)
    parser.add_argument("--k_steps", "-k", type=int, nargs="+",
                        default=[1, 2, 4, 5, 8])
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: runs/<run_name>/nfe_sweep/summary.json)")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        base_conf = load_hyperpyyaml(f)

    run_name = base_conf["env_cfg"].get("run_name", "eval")
    out_path = args.out or "./phase1/nfe_sweep/summary.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {
        "yaml_path": args.yaml_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "results": {},
    }

    for k in args.k_steps:
        print(f"\n{'='*60}\n[NFE sweep] k_steps = {k}\n{'='*60}")
        conf = deepcopy(base_conf)
        override_k_steps(conf, k)
        metrics, run_dir = run_one(conf, k)
        summary["results"][str(k)] = {"metrics": metrics, "run_dir": run_dir}
        print(f"k={k}: " + ", ".join(f"{m}: {v:.4f}" for m, v in metrics.items()))

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable table
    metric_names = list(next(iter(summary["results"].values()))["metrics"].keys())
    print(f"\n\n===== NFE Sweep Summary =====")
    header = "k_steps | " + " | ".join(f"{m:>8}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for k in args.k_steps:
        m = summary["results"][str(k)]["metrics"]
        print(f"{k:>7} | " + " | ".join(f"{m[n]:>8.4f}" for n in metric_names))
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
