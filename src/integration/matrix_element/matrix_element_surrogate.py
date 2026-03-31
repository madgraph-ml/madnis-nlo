import yaml
import torch
import numpy as np

import src.regression as models
import utils.transforms as trans
from src.documenter import Documenter


class SurrogateME:
    """Load a trained MLP surrogate and run inference.

    The model checkpoint now carries the feature standardization stats as
    registered buffers, so no separate means/stds files are needed.
    Calling `net.predict(x)` handles feature standardization internally.
    The caller only needs to apply upstream physics transforms (e.g.
    Np1PhaseSpaceAnglesEnergies) and reverse the amplitude transform
    (LogAmp/LinLogAmp) on the output.
    """

    def __init__(self, folder_run_path: str):
        self.folder_run_path = folder_run_path
        self.doc = Documenter("_", existing_run=folder_run_path, read_only=True)

        with open(folder_run_path + "/final_params.yaml") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        transforms_cfg = params.get("transforms", {})

        # Amplitude transform (LogAmp / LinLogAmp) — needed to reverse the output.
        self.amp_transform = None
        for key in ("LogAmp", "LinLogAmp"):
            if key in transforms_cfg:
                self.amp_transform = getattr(trans, key)(**transforms_cfg[key])
                break

        # Physics feature transforms that run before standardization.
        skip_keys = {"StandardizeFromFile", "StandardizeFromFile_FKS_variant", "LogAmp", "LinLogAmp"}
        self.feature_transforms = [
            getattr(trans, name)(**kwargs)
            for name, kwargs in transforms_cfg.items()
            if name not in skip_keys
        ]

        with torch.no_grad():
            model_cls = getattr(models, params.get("model", "MLP"))
            self.model = model_cls(params, "cpu", self.doc)
            self.model.load(folder_run_path + "/model_best.pt", reg_run=False)

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: raw phase-space features, shape (N, n_features).

        Returns:
            (amplitude, relative_uncertainty) — both as float64 tensors of shape (N,).
        """
        with torch.no_grad():
            x = inputs.detach().cpu().numpy()

            # Apply physics preprocessing (invariants, angles, etc.)
            for t in self.feature_transforms:
                x, _ = t(x, None, None)

            # Forward pass: net.predict() handles feature standardization internally.
            outputs = self.model.net.predict(torch.tensor(x, dtype=torch.float64))

            out0 = outputs[:, 0].detach().cpu().numpy()          # predicted amp (std. space)
            logsigma2 = outputs[:, 1].detach().cpu().numpy()
            uncertainty = np.sqrt(np.exp(logsigma2))
            rel_unc = np.abs(uncertainty / (out0 + 1e-30))

            # De-standardize amplitude using stats baked into the model.
            if self.model.net.standardize:
                amp_std = self.model.net.amp_std.item()
                amp_mean = self.model.net.amp_mean.item()
                out0 = out0 * amp_std + amp_mean

            # Reverse LogAmp / LinLogAmp.
            if self.amp_transform is not None:
                _, out0 = self.amp_transform(None, out0, None, rev=True)

            return (
                torch.tensor(out0, dtype=torch.float64),
                torch.tensor(rel_unc, dtype=torch.float64),
            )
