import utils.transforms as transforms
from documenter import Documenter
import torch


def dtype_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


def get_transformations(transforms_list: dict, doc: Documenter | None = None):
    func = []
    for name, kwargs in transforms_list.items():
        if (
            name == "StandardizeFromFile" or name == "StandardizeFromFile_FKS_variant"
        ) and doc is not None:
            kwargs["model_dir"] = doc.basedir
        func.append(getattr(transforms, name)(**kwargs))
    return func


def clipped_stratified_variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
):
    """
    Computes the stratified variance as introduced in [2311.01548] for two given sets of
    probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
    probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed stratified variance
    """
    if q_sample is None:
        q_sample = q_test
    if channels is None:
        abs_integral = torch.mean(f_true.detach().abs() / q_sample)
        return _clipped_variance(f_true, q_test, q_sample) / abs_integral.square()

    stddev_sum = 0
    abs_integral = 0
    for i in channels.unique():
        mask = channels == i
        fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
        stddev_sum += torch.sqrt(_clipped_variance(fi, qti, qsi) + dtype_epsilon(f_true))
        abs_integral += torch.mean(fi.detach().abs() / qsi)
    return (stddev_sum / abs_integral) ** 2


def _clipped_variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor,
    coeff: torch.Tensor = 50.0,
) -> torch.Tensor:
    """
    Computes the variance for two given sets of probabilities, ``f_true`` and ``q_test``. It uses
    importance sampling with a sampling probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
    Returns:
        computed variance
    """

    def custom_clip(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        return coeff * torch.asinh(x / coeff)

    ratio = q_test / q_sample
    mean = custom_clip(torch.mean(f_true / q_sample), coeff)
    sq = (custom_clip(f_true / q_test, coeff) - mean) ** 2
    return (
        torch.mean(sq * ratio)
        if len(f_true) > 0
        else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
    )
