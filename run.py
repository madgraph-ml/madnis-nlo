import hydra


@hydra.main(config_path="config", config_name="multichannel", version_base=None)
def run(cfg):
    task = str(getattr(cfg, "task", "integrator"))

    if task == "integrator":
        from src.integration.main import main as entry
    elif task == "regression":
        from src.regression.main import main as entry
    else:
        raise ValueError(f"Unknown task='{task}'")

    entry(cfg)


if __name__ == "__main__":
    run()
