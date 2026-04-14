from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
import pytorch_lightning as L
from ELIR.irsetup import IRSetup
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def run_eval(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Prepare datasets
    # ----------------------------
    dataset_cfg = conf.get("dataset_cfg")
    val_dataset = dataset_cfg.get('val_dataset')
    valloader = get_loader(val_dataset)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # ----------------------------
    # Evaluation
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    eval_cfg["log_images"] = True
    eval_cfg.setdefault("max_log_images", 10**6)

    run_name = env_cfg.get("run_name", "eval")
    run_dir = eval_cfg.get("out_folder") or os.path.join("./runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    setup = IRSetup(model, eval_cfg=eval_cfg, run_dir=run_dir)
    trainer = L.Trainer(logger=False)
    set_seed(seed)
    results = trainer.validate(setup, dataloaders=valloader)
    metrics = eval_cfg.get("metrics")
    for metric in metrics:
        print("{}: {:0.4f}".format(metric, results[0][metric]), end =", ")
    print()

    # Save metrics to file
    metrics_dict = {m: float(results[0][m]) for m in metrics}
    record = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": val_dataset,
        "metrics": metrics_dict,
    }
    json_path = os.path.join(run_dir, "eval_metrics.json")
    txt_path = os.path.join(run_dir, "eval_metrics.txt")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    with open(txt_path, "a") as f:
        f.write("[{}] ".format(record["timestamp"]))
        f.write(", ".join("{}: {:0.4f}".format(m, metrics_dict[m]) for m in metrics))
        f.write("\n")
    print("Saved metrics to {} and {}".format(json_path, txt_path))

if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Eval
    # ----------------------------
    run_eval(conf)