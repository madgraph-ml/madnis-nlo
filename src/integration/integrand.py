import torch
import numpy as np
from typing import Union
from src.integration.matrix_element.matrix_element_surrogate import SurrogateME
from src.integration.matrix_element.matrix_element import (
    BornMatrixElement,
    LoopMatrixElement,
    CollinearIntegratedCounterterm,
    SoftIntegratedCounterterm,
    RealEmission,
)
from src.integration.matrix_element.multiprocessing import ThreadedUnaryAPI, ThreadedRealEmissionAPI
from pathlib import Path


INV_GEV2_TO_PB = 0.38937937217186e9
SRC_DIR = Path(__file__).resolve().parent


def from_src(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (SRC_DIR / p).resolve()


class CrossSection:
    def __init__(self, e_cm: float):
        self.e_cm = e_cm

    def __call__(self, momenta: torch.Tensor) -> torch.Tensor:
        return INV_GEV2_TO_PB / (2 * self.e_cm**2) * self.matrix_element(momenta)


class BornCrossSection(CrossSection):
    def __init__(self, e_cm: float, params: dict, device: torch.device):
        super().__init__(e_cm)
        self.api = BornMatrixElement(from_src(params["api_path"]))

    def matrix_element(self, momenta: torch.Tensor) -> torch.Tensor:
        return self.api(momenta)


class CollinearCTCrossSection(CrossSection):
    def __init__(self, e_cm: float, params: dict, device: torch.device):
        super().__init__(e_cm)
        self.api = CollinearIntegratedCounterterm(from_src(params["api_path"]))

    def matrix_element(self, momenta: torch.Tensor) -> torch.Tensor:
        return self.api(momenta)


class NPartCrossSection(CrossSection):
    def __init__(self, e_cm: float, params: dict, device: torch.device):
        super().__init__(e_cm)
        self.api = LoopMatrixElement(from_src(params["api_path"]))

    def matrix_element(self, momenta: torch.Tensor) -> torch.Tensor:
        me = self.api(momenta)
        return me[:, 0] + me[:, 1]


class BornVirtualRatio(CrossSection):
    def __init__(self, e_cm: float, params: dict, device: torch.device):
        super().__init__(e_cm)
        self.api = LoopMatrixElement(from_src(params["api_path"]))
        self.expansion_factor = self.api.alpha_s / (2 * np.pi)

    def matrix_element(self, momenta: torch.Tensor) -> torch.Tensor:
        me = self.api(momenta)
        return me[:, 1] / me[:, 0] / self.expansion_factor


class NLOCrossSection(CrossSection):
    def __init__(
        self, e_cm: float, params: dict, device: torch.device, real_weight: float = 1.0
    ):
        super().__init__(e_cm)
        multithread_cores = params.get("multithread_cores", 4)
        if multithread_cores > 4:
            n_virt = multithread_cores - 3
        else:
            n_virt = 1
        n_real = 1
        n_coll = 1
        n_soft = 1
        print(
            f"NLO Cross Section using {n_virt} virt, {n_coll} coll, {n_soft} soft, {n_real} real threads"
        )
        self.api_born_virt = (
            ThreadedUnaryAPI(
                LoopMatrixElement,
                params["api_path"],
                n_processes=n_virt,
            )
            if multithread_cores > 4
            else LoopMatrixElement(params["api_path"])
        )
        self.api_coll_ct = (
            ThreadedUnaryAPI(
                CollinearIntegratedCounterterm,
                params["api_path_coll"],
                n_processes=n_coll,
            )
            if multithread_cores > 4
            else CollinearIntegratedCounterterm(params["api_path_coll"])
        )
        self.api_soft_ct = (
            ThreadedUnaryAPI(
                SoftIntegratedCounterterm,
                params["api_path_soft"],
                n_processes=n_soft,
            )
            if multithread_cores > 4
            else SoftIntegratedCounterterm(params["api_path_soft"])
        )
        self.api_real_em = (
            ThreadedRealEmissionAPI(
                RealEmission,
                params["api_path_real"],
                n_processes=n_real,
            )
            if multithread_cores > 4
            else RealEmission(params["api_path_real"])
        )
        self.real_weight = real_weight
        self.xi_cut = float(params.get("xi_cut", 0.5))
        self.delta_cut = float(params.get("delta_cut", 1.0))

    def matrix_element(
        self,
        NLO_inputs: list[torch.Tensor],
        pieces: bool = False,
        for_plotting: bool = False,
        madnis_weights: torch.Tensor = None,
        batch_size: int = 10_000,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        b_total = NLO_inputs[0].shape[0]

        if b_total <= batch_size:
            NLO_inputs_local = _clone_mutated_momenta(NLO_inputs)
            return self.matrix_element_impl(
                NLO_inputs_local,
                pieces=pieces,
                for_plotting=for_plotting,
                madnis_weights=madnis_weights,
            )
        # Dict returns (pieces/for_plotting): collect keys batch-wise then stitch
        if pieces or for_plotting:
            out: dict[str] = {}
            first = True
            for start in range(0, b_total, batch_size):
                end = min(start + batch_size, b_total)
                sl = slice(start, end)

                inp_b = _slice_nlo_inputs(NLO_inputs, sl)
                inp_b = _clone_mutated_momenta(inp_b)
                w_b = madnis_weights[sl] if madnis_weights is not None else None

                d = self.matrix_element_impl(
                    inp_b,
                    pieces=pieces,
                    for_plotting=for_plotting,
                    madnis_weights=w_b,
                )

                if first:
                    for k, v in d.items():
                        out[k] = torch.empty(
                            (b_total,) + v.shape[1:],
                            device=v.device,
                            dtype=v.dtype,
                        )
                    first = False

                for k, v in d.items():
                    out[k][sl] = v
                # print every 10%
                if end % max(1, b_total // 10) < batch_size or end == b_total:
                    print(f"  Computed {end} / {b_total} integrands.")

            return out

        dev = NLO_inputs[1][0].device
        dtyp = NLO_inputs[1][0].dtype
        out = torch.empty((b_total,), device=dev, dtype=dtyp)

        for start in range(0, b_total, batch_size):
            end = min(start + batch_size, b_total)
            sl = slice(start, end)

            inp_b = _slice_nlo_inputs(NLO_inputs, sl)
            inp_b = _clone_mutated_momenta(inp_b)
            w_b = madnis_weights[sl] if madnis_weights is not None else None

            out[sl] = self.matrix_element_impl(
                inp_b,
                pieces=False,
                for_plotting=False,
                madnis_weights=w_b,
            )
            # print every 50%
            if end % max(1, b_total // 2) < batch_size or end == b_total:
                print(f"  Computed {end} / {b_total} integrands.")
        return out

    def matrix_element_impl(
        self,
        NLO_inputs: list[torch.Tensor],
        pieces: bool = False,
        for_plotting: bool = False,
        madnis_weights: torch.Tensor = None,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        NLO_inputs: [rad_vars, p_n, p_np1, p_coll, p_soft, k_ct, fks_sec]
        where:
            rad_vars: (b, 3) tensor of radiation variables
            p_n: tuple of (p_n, w_n)
            p_np1: tuple of (p_np1, w_np1)
            p_coll: tuple of (p_coll, w_coll)
            p_soft: tuple of (p_soft, w_soft)
            k_ct: (b, 4) tensor of counterterm momentum
            fks_sec: (b,) tensor of FKS sectors
        """
        rad_vars, p_n, p_np1, p_soft, p_coll, k_ct, fks_sec = NLO_inputs
        xi, y, phi = rad_vars[:, 0], rad_vars[:, 1], rad_vars[:, 2]

        # Born and Virtual contributions
        nbody_mask = p_n[1] != 0
        masked_pn = p_n[0][nbody_mask]

        born_virt_me = torch.zeros(
            (p_n[0].shape[0], 2), device=p_n[0].device, dtype=p_n[0].dtype
        )
        if masked_pn.numel() > 0:
            born_virt_me[nbody_mask] = self.api_born_virt(masked_pn)

        born_me = born_virt_me[:, 0]
        virt_me = born_virt_me[:, 1]

        # Integrated counterterms
        soft_me = torch.zeros_like(p_n[1], device=p_n[1].device, dtype=p_n[1].dtype)
        coll_me = soft_me.clone()
        soft_me[nbody_mask] = self.api_soft_ct(masked_pn, xi_cut=self.xi_cut)
        coll_me[nbody_mask] = self.api_coll_ct(
            masked_pn, xi_cut=self.xi_cut, delta_cut=self.delta_cut
        )

        # Real emission
        real_emission_me = torch.zeros(
            (p_np1[0].shape[0], 4), device=p_np1[0].device, dtype=p_np1[0].dtype
        )
        real_emission_me = self.api_real_em(
            p_np1[0],
            p_n[0],
            p_soft[0],
            p_coll[0],
            k_ct,
            fks_sec,
            xi,
            y,
            xi_cut=self.xi_cut,
            delta_cut=self.delta_cut,
        )

        me_real = torch.zeros_like(p_n[1], device=p_n[1].device, dtype=p_n[1].dtype)
        soft_real = torch.zeros_like(me_real)
        coll_real = torch.zeros_like(me_real)
        soft_coll_real = torch.zeros_like(me_real)
        (
            me_real,
            soft_real,
            coll_real,
            soft_coll_real,
        ) = real_emission_me
        common_factor_nbody = p_n[1] / 6
        born_piece = common_factor_nbody * born_me
        virt_piece = common_factor_nbody * virt_me
        soft_piece = common_factor_nbody * soft_me
        coll_piece = common_factor_nbody * coll_me
        real_piece = (
            (p_np1[1] * me_real - p_coll[1] * coll_real) / (xi * (1 - y))
            - p_soft[1] * soft_real
            + p_soft[1] * soft_coll_real
        ) * self.real_weight
        if pieces:
            return {
                "born_me": born_me,
                "virt_me": virt_me,
                "soft_me": soft_me,
                "coll_me": coll_me,
                "real_me": me_real,
                "coll_ct": coll_real,
                "soft_ct": soft_real,
                "soft_coll_ct": soft_coll_real,
                "total": born_piece + virt_piece + soft_piece + coll_piece + real_piece,
            }
        if for_plotting:
            cs_prefactor = INV_GEV2_TO_PB / (2 * self.e_cm**2)
            madnis_weights_without_integrand = 1.0 / madnis_weights
            w = cs_prefactor * madnis_weights_without_integrand
            born_weight = w * common_factor_nbody * born_me
            virt_weight = w * common_factor_nbody * virt_me
            soft_weight = w * common_factor_nbody * soft_me
            coll_weight = w * common_factor_nbody * coll_me
            real_weight = w * p_np1[1] * me_real / (xi * (1 - y))
            local_soft_weight = w * (-p_soft[1] * soft_real)
            local_coll_weight = w * (-p_coll[1] * coll_real) / (xi * (1 - y))
            local_soft_coll_weight = w * (p_soft[1] * soft_coll_real)
            total_weight = (
                born_weight
                + virt_weight
                + soft_weight
                + coll_weight
                + real_weight
                + local_soft_weight
                + local_coll_weight
                + local_soft_coll_weight
            )
            out_dictionary = {
                "fks_sector": fks_sec,
                "nbody_weight": (born_weight + virt_weight + soft_weight + coll_weight),
                "real_weight": real_weight,
                "total_weight": total_weight,
            }
            return out_dictionary
        return born_piece + virt_piece + soft_piece + coll_piece + real_piece

    def close(self):
        for name in ["api_born_virt", "api_coll_ct", "api_soft_ct", "api_real_em"]:
            api = getattr(self, name, None)
            if hasattr(api, "close"):
                api.close()


class NLOSurrogateCrossSection(CrossSection):
    def __init__(
        self, e_cm: float, params: dict, device: torch.device, real_weight: float = 1.0
    ):
        super().__init__(e_cm)
        multithread_cores = params.get("multithread_cores", 5)
        if multithread_cores > 5:
            n_virt = multithread_cores - 4
        else:
            n_virt = 1
        n_real = 1
        n_coll = 1
        n_soft = 1
        n_born = 1
        print(
            f"NLO Cross Section using {n_born} born, {n_virt} virt, {n_coll} coll, {n_soft} soft, {n_real} real threads"
        )
        self.real_weight = real_weight
        self.xi_cut = float(params.get("xi_cut", 0.5))
        self.delta_cut = float(params.get("delta_cut", 1.0))
        self.surrogate_section = params.get("surrogate_paths", None)
        self._r_surr_used = 0
        self._r_surr_total = 0
        self._r_surr_calls = 0
        self.true_me = params.get("true_me", None)
        self.contributions_dictionary = {
            "born": True,
            "virtual": True,
            "icts": True,
            "real": True,
            "V_surr": False,
            "R_surr": False,
        }
        if self.surrogate_section is not None:
            for k, v in self.surrogate_section.items():
                setattr(self, k, SurrogateME(v))
        if self.true_me is not None:
            for k, v in self.true_me.items():
                assert k in ["born_me", "loop_me", "soft_me", "coll_me", "real_me"]
                if v is not None:
                    if k == "born_me":
                        self.born_me = (
                            ThreadedUnaryAPI(BornMatrixElement, v, n_processes=n_born)
                            if n_born > 1
                            else BornMatrixElement(v)
                        )
                    if k == "loop_me":
                        self.loop_me = (
                            ThreadedUnaryAPI(LoopMatrixElement, v, n_processes=n_virt)
                            if n_virt > 1
                            else LoopMatrixElement(v)
                        )
                    if k == "soft_me":
                        self.soft_me = (
                            ThreadedUnaryAPI(
                                SoftIntegratedCounterterm, v, n_processes=n_soft
                            )
                            if n_soft > 1
                            else SoftIntegratedCounterterm(v)
                        )
                    if k == "coll_me":
                        self.coll_me = (
                            ThreadedUnaryAPI(
                                CollinearIntegratedCounterterm,
                                v,
                                n_processes=n_coll,
                            )
                            if n_coll > 1
                            else CollinearIntegratedCounterterm(v)
                        )
                    if k == "real_me":
                        self.real_me = (
                            ThreadedRealEmissionAPI(RealEmission, v, n_processes=n_real)
                            if n_real > 1
                            else RealEmission(v)
                        )

    @property
    def r_surr_usage_fraction(self) -> float:
        """Fraction of points (across all matrix_element_impl calls) where the
        real surrogate was used instead of the true ME."""
        if self._r_surr_total == 0:
            return 0.0
        return self._r_surr_used / self._r_surr_total

    def _r_surr_config(self) -> tuple[bool, float, float]:
        """Return (enabled, xi_threshold, y_threshold) for the real surrogate.

        Thresholds are derived from the FKS singularity cuts stored on self:
          xi_threshold = xi_cut        (surrogate used for xi >= xi_cut)
          y_threshold  = 1 - delta_cut (surrogate used for y  <= 1 - delta_cut)

        R_surr in contributions_dictionary controls only whether it is enabled
        (plain bool or dict with an "enabled" key).
        """
        r_surr = self.contributions_dictionary.get("R_surr", False)
        enabled = bool(r_surr.get("enabled", False)) if isinstance(r_surr, dict) else bool(r_surr)
        return enabled, self.xi_cut, 1.0 - self.delta_cut

    def matrix_element(
        self,
        NLO_inputs: list[torch.Tensor],
        pieces: bool = False,
        for_plotting: bool = False,
        madnis_weights: torch.Tensor = None,
        batch_size: int = 10_000,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        b_total = NLO_inputs[0].shape[0]

        if b_total <= batch_size:
            NLO_inputs_local = _clone_mutated_momenta(NLO_inputs)
            return self.matrix_element_impl(
                NLO_inputs_local,
                pieces=pieces,
                for_plotting=for_plotting,
                madnis_weights=madnis_weights,
            )
        # Dict returns (pieces/for_plotting): collect keys batch-wise then stitch
        if pieces or for_plotting:
            out: dict[str] = {}
            first = True
            for start in range(0, b_total, batch_size):
                end = min(start + batch_size, b_total)
                sl = slice(start, end)

                inp_b = _slice_nlo_inputs(NLO_inputs, sl)
                inp_b = _clone_mutated_momenta(inp_b)
                w_b = madnis_weights[sl] if madnis_weights is not None else None

                d = self.matrix_element_impl(
                    inp_b,
                    pieces=pieces,
                    for_plotting=for_plotting,
                    madnis_weights=w_b,
                )

                if first:
                    # allocate full-size containers
                    for k, v in d.items():
                        if torch.is_tensor(v):
                            out[k] = torch.empty(
                                (b_total,) + v.shape[1:],
                                device=v.device,
                                dtype=v.dtype,
                            )
                        else:
                            out[k] = torch.empty(
                                (b_total,) + v.shape[1:],
                                device=v.device,
                                dtype=v.dtype,
                            )
                    first = False

                for k, v in d.items():
                    out[k][sl] = v
                # print every 10%
                if end % max(1, b_total // 10) < batch_size or end == b_total:
                    print(f"  Computed {end} / {b_total} integrands.")

            return out

        dev = NLO_inputs[1][0].device
        dtyp = NLO_inputs[1][0].dtype
        out = torch.empty((b_total,), device=dev, dtype=dtyp)

        for start in range(0, b_total, batch_size):
            end = min(start + batch_size, b_total)
            sl = slice(start, end)

            inp_b = _slice_nlo_inputs(NLO_inputs, sl)
            inp_b = _clone_mutated_momenta(inp_b)
            w_b = madnis_weights[sl] if madnis_weights is not None else None

            out[sl] = self.matrix_element_impl(
                inp_b,
                pieces=False,
                for_plotting=False,
                madnis_weights=w_b,
            )
            # print every 50%
            if end % max(1, b_total // 2) < batch_size or end == b_total:
                print(f"  Computed {end} / {b_total} integrands.")
        return out

    def matrix_element_impl(
        self,
        NLO_inputs: list[torch.Tensor],
        pieces: bool = False,
        for_plotting: bool = False,
        madnis_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        NLO_inputs: [rad_vars, nbody, np1body, coll, soft, k_ct, fks_sec]
        where:
            rad_vars: (b, 3) tensor of radiation variables
            nbody: tuple of (p_n, w_n)
            np1body: tuple of (p_np1, w_np1)
            coll: tuple of (p_coll, w_coll)
            soft: tuple of (p_soft, w_soft)
            k_ct: (b, 4) tensor of counterterm momentum
            fks_sec: (b,) tensor of FKS sectors
        """
        rad_vars, p_n, p_np1, p_soft, p_coll, k_ct, fks_sec = NLO_inputs
        nbody_mask = p_n[1] != 0
        masked_pn = p_n[0][nbody_mask]
        b_size = p_n[0].shape[0]
        dev = p_n[0].device
        dtyp = p_n[0].dtype
        xi = rad_vars[:, 0]
        y = rad_vars[:, 1]
        phi = rad_vars[:, 2]
        ## np1body section
        r_surr_enabled, xi_thr, y_thr = self._r_surr_config()
        if (
            r_surr_enabled or for_plotting
        ) and self.contributions_dictionary["real"]:
            with torch.no_grad():
                real_m = p_np1[0].reshape(p_np1[0].shape[0], -1)
                soft_m = p_soft[0].reshape(p_np1[0].shape[0], -1)
                coll_m = p_coll[0].reshape(p_np1[0].shape[0], -1)
                xi = rad_vars[:, 0].unsqueeze(-1)
                y = rad_vars[:, 1].unsqueeze(-1)
                phi = rad_vars[:, 2].unsqueeze(-1)
                fks_sec_inp = fks_sec.unsqueeze(-1)
                soft_mask = torch.where(
                    (xi.squeeze() < xi_thr)
                    & ((fks_sec == 1) | (fks_sec == 2) | (fks_sec == 3)),
                    1.0,
                    0.0,
                )
                coll_mask = torch.where((y.squeeze() > y_thr), 1.0, 0.0)
                softcoll_mask = torch.where(
                    (xi.squeeze() < xi_thr)
                    & (y.squeeze() > y_thr)
                    & ((fks_sec == 1) | (fks_sec == 2) | (fks_sec == 3)),
                    1.0,
                    0.0,
                )

                if r_surr_enabled and not for_plotting:
                    # surr_mask: points where xi >= xi_thr AND y <= y_thr (away from
                    # singularities). On these points soft/coll/softcoll masks are all 0,
                    # so only me_real from the surrogate contributes.
                    surr_mask = ~((xi.squeeze() < xi_thr) | (y.squeeze() > y_thr))
                    real_mask = ~surr_mask

                    me_real = torch.zeros(b_size, device=dev, dtype=dtyp)
                    soft_real = torch.zeros(b_size, device=dev, dtype=dtyp)
                    coll_real = torch.zeros(b_size, device=dev, dtype=dtyp)
                    soft_coll_real = torch.zeros(b_size, device=dev, dtype=dtyp)
                    # Evaluate true ME only on points outside the surrogate region
                    if real_mask.any():
                        real_em_sub = self.real_me(
                            p_np1[0][real_mask],
                            p_n[0][real_mask],
                            p_soft[0][real_mask],
                            p_coll[0][real_mask],
                            k_ct[real_mask],
                            fks_sec[real_mask],
                            xi.squeeze()[real_mask],
                            y.squeeze()[real_mask],
                            xi_cut=self.xi_cut,
                            delta_cut=self.delta_cut,
                        )
                        me_real[real_mask] = real_em_sub[0]
                        soft_real[real_mask] = real_em_sub[1]
                        coll_real[real_mask] = real_em_sub[2]
                        soft_coll_real[real_mask] = real_em_sub[3]

                    # Evaluate surrogate only on points inside the surrogate region.
                    # soft/coll/softcoll contributions are 0 there, so only me_real needed.
                    if surr_mask.any():
                        try:
                            surr_r = self.real_surrogate(
                                torch.cat(
                                    (
                                        real_m[surr_mask],
                                        k_ct[surr_mask],
                                        xi[surr_mask],
                                        y[surr_mask],
                                        phi[surr_mask],
                                        fks_sec_inp[surr_mask],
                                    ),
                                    dim=-1,
                                )
                            )
                            me_surr = surr_r[0].detach().clone().squeeze()
                            me_real[surr_mask] = torch.nan_to_num(
                                me_surr, nan=0.0, posinf=0.0, neginf=0.0
                            )
                        except:
                            pass  # me_real[surr_mask] remains 0

                    n_used = int(surr_mask.sum().item())
                    n_total = b_size
                    self._r_surr_used += n_used
                    self._r_surr_total += n_total
                    self._r_surr_calls += 1
                    if self._r_surr_calls % 10 == 1:
                        print(
                            f"  Real surrogate usage: {n_used / n_total:.3f}"
                            f" (running avg: {self.r_surr_usage_fraction:.3f})"
                        )

                else:
                    # for_plotting: evaluate surrogate and real ME on the full batch
                    # so that both can be compared in plots.
                    real_emission_me = self.real_me(
                        p_np1[0],
                        p_n[0],
                        p_soft[0],
                        p_coll[0],
                        k_ct,
                        fks_sec,
                        xi.squeeze(),
                        y.squeeze(),
                        xi_cut=self.xi_cut,
                        delta_cut=self.delta_cut,
                    )
                    try:
                        surr_r = self.real_surrogate(
                            torch.cat((real_m, k_ct, xi, y, phi, fks_sec_inp), dim=-1)
                        )
                    except:
                        surr_r = [torch.zeros(xi.squeeze().shape[0])]
                    # =========================
                    # SOFT REAL
                    # =========================
                    try:
                        surr_soft = self.real_surrogate(
                            torch.cat(
                                (coll_m, k_ct, torch.zeros_like(xi), y, phi, fks_sec_inp),
                                dim=-1,
                            )
                        )
                    except:
                        surr_soft = [torch.zeros(xi.squeeze().shape[0])]
                    # =========================
                    # COLL REAL
                    # =========================
                    try:
                        surr_coll = self.real_surrogate(
                            torch.cat(
                                (coll_m, k_ct, xi, torch.ones_like(y), phi, fks_sec_inp),
                                dim=-1,
                            )
                        )
                    except:
                        surr_coll = [torch.zeros(xi.squeeze().shape[0])]
                    # =========================
                    # SOFT COLL REAL
                    # =========================
                    try:
                        surr_softcoll = self.real_surrogate(
                            torch.cat(
                                (
                                    soft_m,
                                    k_ct,
                                    torch.zeros_like(xi),
                                    torch.ones_like(y),
                                    phi,
                                    fks_sec_inp,
                                ),
                                dim=-1,
                            )
                        )
                    except:
                        surr_softcoll = [torch.zeros(xi.squeeze().shape[0])]

                    me_real_surrogate = torch.nan_to_num(
                        surr_r[0].detach().clone().squeeze(),
                        nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    soft_real_surrogate = torch.nan_to_num(
                        surr_soft[0].detach().clone().squeeze() * soft_mask.squeeze(),
                        nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    coll_real_surrogate = torch.nan_to_num(
                        surr_coll[0].detach().clone().squeeze() * coll_mask.squeeze(),
                        nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    soft_coll_real_surrogate = torch.nan_to_num(
                        surr_softcoll[0].detach().clone().squeeze() * softcoll_mask.squeeze(),
                        nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    global_unc_mask = (y.squeeze() > y_thr) | (xi.squeeze() < xi_thr)
                    me_real_surrogate = torch.where(
                        global_unc_mask, real_emission_me[0], me_real_surrogate
                    )
                    soft_real_surrogate = torch.where(
                        global_unc_mask, real_emission_me[1], soft_real_surrogate
                    )
                    coll_real_surrogate = torch.where(
                        global_unc_mask, real_emission_me[2], coll_real_surrogate
                    )
                    soft_coll_real_surrogate = torch.where(
                        global_unc_mask, real_emission_me[3], soft_coll_real_surrogate
                    )
                    me_real, soft_real, coll_real, soft_coll_real = real_emission_me

        elif self.contributions_dictionary["real"]:
            real_emission_me = torch.zeros(
                (p_np1[0].shape[0], 4), device=p_np1[0].device, dtype=p_np1[0].dtype
            )
            if self.contributions_dictionary["real"]:
                real_emission_me = self.real_me(
                    p_np1[0], p_n[0], p_soft[0], p_coll[0], k_ct, fks_sec, xi, y,
                    xi_cut=self.xi_cut, delta_cut=self.delta_cut,
                )
            me_real, soft_real, coll_real, soft_coll_real = real_emission_me
        else:
            me_real = torch.zeros((b_size,), device=dev, dtype=dtyp)
            soft_real = torch.zeros((b_size,), device=dev, dtype=dtyp)
            coll_real = torch.zeros((b_size,), device=dev, dtype=dtyp)
            soft_coll_real = torch.zeros((b_size,), device=dev, dtype=dtyp)
        if nbody_mask.sum() > 0:
            masked_pn_nn_inp = masked_pn.reshape(masked_pn.shape[0], -1)
        xi = rad_vars[:, 0]
        y = rad_vars[:, 1]
        phi = rad_vars[:, 2]
        if not self.contributions_dictionary["V_surr"]:
            born_virt_me = torch.zeros((b_size, 2), device=dev, dtype=dtyp)
            soft_me = torch.zeros((b_size,), device=dev, dtype=dtyp)
            coll_me = torch.zeros((b_size,), device=dev, dtype=dtyp)
            if self.contributions_dictionary["virtual"]:
                born_virt_me[nbody_mask] = self.loop_me(masked_pn)
            elif self.contributions_dictionary["born"]:
                born_virt_me[nbody_mask, 0] = self.born_me(masked_pn)
            born_me = born_virt_me[:, 0]
            virt_me = born_virt_me[:, 1]
            if self.contributions_dictionary["icts"]:
                soft_me[nbody_mask] = self.soft_me(masked_pn, xi_cut=self.xi_cut)
                coll_me[nbody_mask] = self.coll_me(
                    masked_pn, xi_cut=self.xi_cut, delta_cut=self.delta_cut
                )
            if for_plotting:
                v_b_ratio = torch.zeros((b_size,), device=dev, dtype=dtyp)
                if (nbody_mask.sum() > 0) and self.contributions_dictionary["virtual"]:
                    v_b_ratio[nbody_mask] = self.v_b_ratio(masked_pn_nn_inp)[0].detach()
                virt_me_surrogate = torch.zeros((b_size,), device=dev, dtype=dtyp)
                virt_me_surrogate[nbody_mask] = (
                    born_me[nbody_mask] * v_b_ratio[nbody_mask]
                )

        elif self.contributions_dictionary["V_surr"]:
            v_b_ratio = torch.zeros((b_size,), device=dev, dtype=dtyp)
            born_me = torch.zeros((b_size,), device=dev, dtype=dtyp)
            if (nbody_mask.sum() > 0) and self.contributions_dictionary["virtual"]:
                v_b_ratio[nbody_mask] = self.v_b_ratio(masked_pn_nn_inp)[0].detach()
            if self.contributions_dictionary["born"]:
                born_me[nbody_mask] = self.born_me(masked_pn)
            virt_me = born_me * v_b_ratio
            soft_me = torch.zeros((b_size,), device=dev, dtype=dtyp)
            coll_me = torch.zeros((b_size,), device=dev, dtype=dtyp)
            soft_me[nbody_mask] = self.soft_me(masked_pn, xi_cut=self.xi_cut)
            coll_me[nbody_mask] = self.coll_me(
                masked_pn, xi_cut=self.xi_cut, delta_cut=self.delta_cut
            )
            if for_plotting:
                virt_me_surrogate = virt_me.clone()
                born_virt_me = torch.zeros((b_size, 2), device=dev, dtype=dtyp)
                if self.contributions_dictionary["virtual"]:
                    born_virt_me[nbody_mask] = self.loop_me(masked_pn)
                elif self.contributions_dictionary["born"]:
                    born_virt_me[nbody_mask, 0] = self.born_me(masked_pn)
                born_me = born_virt_me[:, 0]
                virt_me = born_virt_me[:, 1]
        else:
            raise ValueError(
                f"Unknown nbody_surrogate_identifier: {self.nbody_surrogate_identifier}"
            )

        common_factor_nbody = p_n[1] / 6
        born_piece = common_factor_nbody * born_me
        virt_piece = common_factor_nbody * virt_me
        soft_piece = common_factor_nbody * soft_me
        coll_piece = common_factor_nbody * coll_me
        real_piece = (
            (p_np1[1] * (me_real) - p_coll[1] * (coll_real)) / (xi * (1 - y))
            - p_soft[1] * (soft_real)
            + p_soft[1] * (soft_coll_real)
        )
        o = (
            born_piece
            + virt_piece * self.real_weight
            + soft_piece * self.real_weight
            + coll_piece * self.real_weight
            + real_piece * self.real_weight
        )
        o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
        if pieces:
            return {
                "rad_vars": rad_vars,
                "p_n": p_n,
                "p_np1": p_np1,
                "p_soft": p_soft,
                "p_coll": p_coll,
                "p_soft_ct": k_ct,
                "fks_sector": fks_sec,
                "born_me": born_me,
                "virt_me": virt_me,
                "soft_me": soft_me,
                "coll_me": coll_me,
                "real_me": me_real,
                "coll_ct": coll_real,
                "soft_ct": soft_real,
                "soft_coll_ct": soft_coll_real,
                "total": born_piece + virt_piece + soft_piece + coll_piece + real_piece,
            }
        if for_plotting:
            cs_prefactor = INV_GEV2_TO_PB / (2 * self.e_cm**2)
            madnis_weights_without_integrand = 1.0 / madnis_weights
            w = cs_prefactor * madnis_weights_without_integrand
            born_weight = w * common_factor_nbody * born_me
            virt_weight = w * common_factor_nbody * virt_me
            virt_weight_surrogate = w * common_factor_nbody * virt_me_surrogate
            soft_weight = w * common_factor_nbody * soft_me
            coll_weight = w * common_factor_nbody * coll_me
            real_weight = w * p_np1[1] * me_real / (xi * (1 - y))
            local_soft_weight = w * (-p_soft[1] * soft_real)
            local_coll_weight = w * (-p_coll[1] * coll_real) / (xi * (1 - y))
            local_soft_coll_weight = w * (p_soft[1] * soft_coll_real)
            real_weight_surrogate = w * p_np1[1] * me_real_surrogate / (xi * (1 - y))
            local_soft_weight_surrogate = w * (-p_soft[1] * soft_real_surrogate)
            local_coll_weight_surrogate = (
                w * (-p_coll[1] * coll_real_surrogate) / (xi * (1 - y))
            )
            local_soft_coll_weight_surrogate = w * (p_soft[1] * soft_coll_real_surrogate)
            total_weight = (
                born_weight
                + virt_weight
                + soft_weight
                + coll_weight
                + real_weight
                + local_soft_weight
                + local_coll_weight
                + local_soft_coll_weight
            )
            out_dictionary = {
                "fks_sector": fks_sec,
                "nbody_weight": (
                    born_weight
                    + virt_weight
                    + soft_weight
                    + coll_weight
                    + local_soft_weight
                    + local_coll_weight
                    + local_soft_coll_weight
                ),
                "nbody_weight_V_surrogate": (
                    born_weight
                    + virt_weight_surrogate
                    + soft_weight
                    + coll_weight
                    + local_soft_weight
                    + local_coll_weight
                    + local_soft_coll_weight
                ),
                "nbody_weight_VR_surrogate": (
                    born_weight
                    + virt_weight_surrogate
                    + soft_weight
                    + coll_weight
                    + local_soft_weight_surrogate
                    + local_coll_weight_surrogate
                    + local_soft_coll_weight_surrogate
                ),
                "real_weight": real_weight,
                "real_weight_surrogate": real_weight_surrogate,
                "real_subtracted_weight": local_soft_weight+local_coll_weight+local_soft_coll_weight+real_weight,
                "total_weight": total_weight,
            }
            return out_dictionary
        return o

    def close(self):
        for name in ["born_me", "loop_me", "soft_me", "coll_me", "real_me"]:
            api = getattr(self, name, None)
            if hasattr(api, "close"):
                api.close()


def _slice_nlo_inputs(NLO_inputs, sl: slice):
    rad_vars, p_n, p_np1, p_soft, p_coll, k_ct, fks_sec = NLO_inputs

    def slice_tensor(x):
        return x[sl] if x is not None else None

    def slice_pair(pair):
        if pair is None:
            return None
        mom, w = pair
        return (mom[sl], w[sl])

    return [
        slice_tensor(rad_vars),
        slice_pair(p_n),
        slice_pair(p_np1),
        slice_pair(p_soft),
        slice_pair(p_coll),
        slice_tensor(k_ct),
        slice_tensor(fks_sec),
    ]


def _clone_mutated_momenta(NLO_inputs_batch):
    """Clone only the tensors you later write into (momenta arrays)."""
    rad_vars, p_n, p_np1, p_soft, p_coll, k_ct, fks_sec = NLO_inputs_batch

    def clone_pair(pair):
        if pair is None:
            return None
        mom, w = pair
        return (mom.clone(), w)  # clone momenta only

    return [
        rad_vars,  # read-only
        clone_pair(p_n),  # mutated in permutation block
        clone_pair(p_np1),  # mutated in permutation block
        clone_pair(p_soft),  # mutated in permutation block
        clone_pair(p_coll),  # mutated in permutation block
        k_ct,  # read-only
        fks_sec,  # read-only
    ]


def apply_uncertainty_gate(value, uncertainty, threshold=0.0001):
    ratio = torch.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    return ratio > threshold
