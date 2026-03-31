import torch
import numpy as np


def tensors_to_numpy(tensors):
    """
    Convert a list (or nested list) of np.ndarrays to NumPy arrays.
    Detaches, moves to CPU, preserves shapes.
    """
    return [
        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in tensors
    ]


def delta_R(phi1, eta1, phi2, eta2) -> np.ndarray:
    deta = delta_eta(eta1, eta2, abs=False)
    dphi = delta_phi(phi1, phi2, abs=False)
    return np.sqrt(deta**2 + dphi**2)


def delta_eta(eta1: np.ndarray, eta2: np.ndarray, abs: bool = True) -> np.ndarray:
    deta = eta1 - eta2
    return np.abs(deta) if abs else deta


def delta_phi(phi1: np.ndarray, phi2: np.ndarray, abs: bool = True) -> np.ndarray:
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.abs(dphi) if abs else dphi


def get_pt(particle):
    return np.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return np.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle, eps=1e-10):
    p_abs = np.sqrt(np.sum(particle[..., 1:] ** 2, axis=-1))
    eta = 0.5 * (
        np.log(np.clip(np.abs(p_abs + particle[..., 3]), eps, None))
        - np.log(np.clip(np.abs(p_abs - particle[..., 3]), eps, None))
    )
    return eta


def get_mass(particle, eps=1e-6):
    return np.sqrt(
        np.clip(
            particle[..., 0] ** 2 - np.sum(particle[..., 1:] ** 2, axis=-1),
            eps,
            None,
        )
    )


def jet_pt(jets):
    return [get_pt(jet) for jet in jets]


def EPxPyPz_to_PtPhiEtaM(particles: np.ndarray, eps: float = 1e-10):
    pt = get_pt(particles)
    phi = get_phi(particles)
    eta = get_eta(particles, eps=eps)
    m = get_mass(particles, eps=eps)

    return np.stack((pt, phi, eta, m), axis=-1)


def list_EPxPyPz_to_PtPhiEtaM(jets_list, eps: float = 1e-10):
    return [EPxPyPz_to_PtPhiEtaM(j, eps) for j in jets_list]


def PtPhiEtaM_to_EPxPyPz(PtPhiEtaM, cutoff=10.0):

    if PtPhiEtaM.shape[-1] == 4:
        pt = PtPhiEtaM[..., 0]
        phi = PtPhiEtaM[..., 1]
        eta = PtPhiEtaM[..., 2]
        mass = PtPhiEtaM[..., 3]
    elif PtPhiEtaM.shape[-1] == 3:
        pt = PtPhiEtaM[..., 0]
        phi = PtPhiEtaM[..., 1]
        eta = PtPhiEtaM[..., 2]
        mass = np.zeros_like(pt)
    else:
        raise ValueError(f"PtPhiEtaM has wrong shape {PtPhiEtaM.shape}")

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -cutoff, cutoff))
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)

    EPxPyPz = np.stack((E, px, py, pz), axis=-1)

    assert np.isfinite(EPxPyPz).all(), (
        f"NaNs: {np.isnan(EPxPyPz).sum(axis=0)}, "
        f"Infs: {np.isinf(EPxPyPz).sum(axis=0)}"
    )

    return EPxPyPz


def list_PtPhiEtaM_to_EPxPyPz(jets_list, cutoff=10.0):
    return [PtPhiEtaM_to_EPxPyPz(j, cutoff=cutoff) for j in jets_list]


def sort_by_pt(jets):
    "Takes in input a list of jets expressed in PtEtaPhiM"
    return [jet[np.argsort(-jet[:, 0])] for jet in jets]
