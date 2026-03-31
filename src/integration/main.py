import os
import traceback

import hydra
import numpy as np
import shutil
import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from src.documenter import Documenter
from src.integration.integrator import ParxIntegrator


def _archive_config_original(run_dir: str) -> None:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return

    old_dir = os.path.join(run_dir, "old")
    os.makedirs(old_dir, exist_ok=True)

    dst = os.path.join(old_dir, "config_0.yaml")
    if not os.path.exists(dst):
        os.rename(cfg_path, dst)
        return

    k = 1
    while True:
        dst_k = os.path.join(old_dir, f"config_{k}.yaml")
        if not os.path.exists(dst_k):
            os.rename(cfg_path, dst_k)
            return
        k += 1


def _set_torch_dtype(dtype: str | None) -> None:
    if not dtype:
        return
    if dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif dtype == "float16":
        torch.set_default_dtype(torch.float16)
    else:
        raise ValueError(f"Unknown dtype '{dtype}'")


def _resolve_device(cfg: DictConfig) -> str:
    use_cuda = (
        bool(getattr(cfg, "device", {}).get("use_cuda", False))
        and torch.cuda.is_available()
    )
    which = int(getattr(cfg, "device", {}).get("which_cuda", 0))
    return f"cuda:{which}" if use_cuda else "cpu"


def _rotate_run_dir_contents(run_dir: str) -> None:
    old_root = os.path.join(run_dir, "old")
    os.makedirs(old_root, exist_ok=True)

    # next index inside old/
    existing = [
        int(d)
        for d in os.listdir(old_root)
        if d.isdigit() and os.path.isdir(os.path.join(old_root, d))
    ]
    next_idx = max(existing) + 1 if existing else 0
    archive_dir = os.path.join(old_root, str(next_idx))
    os.makedirs(archive_dir, exist_ok=True)

    for item in os.listdir(run_dir):
        if item == "old":
            continue  # never move old/ itself

        item_path = os.path.join(run_dir, item)

        keep = (
            item == "integrator.pth"
            or (
                os.path.isfile(item_path)
                and item.startswith("log")
                and item.endswith(".txt")
            )
            or (
                os.path.isfile(item_path)
                and item.startswith("out_")
                and item.endswith(".log")
            )
        )
        if keep:
            continue

        shutil.move(item_path, os.path.join(archive_dir, item))


def _make_new_run_dir(base_dir: str, cfg: DictConfig) -> str:
    """
    Create a new run directory under <repo>/results/<timestamp[-name]>.
    """
    process = cfg.get("process", "unknown")

    results_dir = os.path.join(base_dir, "results", process)
    os.makedirs(results_dir, exist_ok=True)

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    if cfg.run.name is not None:
        run_name = f"{stamp}-{cfg.run.name}"
    else:
        run_name = stamp

    return os.path.join(results_dir, run_name)


def _save_config_and_snapshot_src(base_dir: str, run_dir: str, cfg: DictConfig) -> None:
    """
    Save a config.yaml into run_dir, but with run.type forced to 'plot' and run.path=run_dir,
    then snapshot src/ into run_dir/src.
    """
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)

    cfg_to_save.setdefault("run", {})
    cfg_to_save["run"]["path"] = run_dir
    cfg_to_save["run"]["type"] = "plot"

    config_file = os.path.join(run_dir, "config.yaml")
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(cfg_to_save)))

    shutil.copytree(
        os.path.join(base_dir, "src"),
        os.path.join(run_dir, "src"),
        dirs_exist_ok=True,
        symlinks=False,
        ignore_dangling_symlinks=True,
        ignore=shutil.ignore_patterns(
            "*__pycache__",
            "*egg-info",
            "playground",
            "template_files",
            "*__init__.py",
            "process_api_storage",
        ),
    )


def run(run_dir: str, cfg: DictConfig, doc: Documenter) -> None:
    """
    Main run function
    """
    _set_torch_dtype(cfg.get("dtype", None))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    seed = cfg.get("seed", None)
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print(f"Starting run in {run_dir}")
    device = _resolve_device(cfg)
    params = OmegaConf.to_container(cfg, resolve=True)
    integrator = ParxIntegrator(params["integrator"], device, doc)
    s = (
        "Vegas pretraining"
        if params["integrator"].get("vegas_pretraining", False)
        else "Vegas training"
        if params["integrator"].get("vegas", False)
        else None
    )
    if not s:
        s = (
            "MadNIS training"
            if params["integrator"].get("train_iterations", 0) > 0
            else "Integration"
        )
    else:
        s += (
            " + MadNIS training"
            if params["integrator"].get("train_iterations", 0) > 0
            else ""
        )
    print(f"Integrating process {params.get('process', 'N/A')}")
    print(f"Mode: {s}" if cfg.run.type == "train" else "Plotting mode")
    print(f"device: {device}")
    print(f"dtype: {torch.get_default_dtype()}")

    # replicate your old behavior: train only when run.type == "train"
    if cfg.run.type == "train":
        if "nlo" in integrator.cross_sections:
            integrator.cross_sections["nlo"].contributions_dictionary = params[
                "integrator"
            ]["train_contributions"]
        integrator.train()
        print(f"Saving flow model at {run_dir}/integrator.pth")
        torch.save(integrator.integrator.state_dict(), run_dir + "/integrator.pth")
        print(f"Flow model saved!")
    elif cfg.run.type == "plot":
        model_path = os.path.join(cfg.run.path, "integrator.pth")
        integrator.integrator.load_state_dict(torch.load(model_path))
        print(f"Loaded integrator from {model_path}")

    eval_mode = str(cfg.integrator.eval_mode)
    if "nlo" in integrator.cross_sections:
        integrator.cross_sections["nlo"].contributions_dictionary = params["integrator"][
            "eval_contributions"
        ]
    if eval_mode == "dataset":
        integrator.generate_dataset()
    elif eval_mode == "integrate":
        integrator.integrate()
    else:
        raise ValueError(f"Unknown eval mode '{eval_mode}'")

    integrator.make_plots()

    # Closing API threads
    for cs in integrator.cross_sections.values():
        if hasattr(cs, "close"):
            cs.close()
    print("Run finished")


@hydra.main(version_base="1.3", config_path="config", config_name="integrator_run")
def main(cfg: DictConfig) -> None:
    """
    Main entry point
    Creates folders and snapshots src. Calls main run() function
    """
    base_dir = hydra.utils.get_original_cwd()

    if cfg.run.type == "train":
        if not cfg.warm_start:
            run_dir = _make_new_run_dir(base_dir, cfg)
            os.makedirs(run_dir, exist_ok=True)
        else:
            raise NotImplementedError("Warm start not implemented yet")

        _save_config_and_snapshot_src(base_dir, run_dir, cfg)

    elif cfg.run.type == "plot":
        if not cfg.run.path:
            raise ValueError("run.path must be set when run.type=plot")
        run_dir = os.path.abspath(os.path.join(base_dir, str(cfg.run.path)))
        os.makedirs(run_dir, exist_ok=True)
        _rotate_run_dir_contents(run_dir)
        _archive_config_original(run_dir)
        _save_config_and_snapshot_src(base_dir, run_dir, cfg)

    else:
        raise NotImplementedError(
            f"Run type {cfg.run.type} not recognized (expected train|plot)"
        )

    doc = Documenter(
        str(cfg.run.name) if cfg.run.name is not None else "run",
        existing_run=run_dir,
    )
    try:
        run(run_dir, cfg, doc)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
