import re
from pathlib import Path

import torch
from torchspace.rambo import RamboOnDiet
from torchspace.cuts import PhaseSpaceCuts

# FKS soft/collinear region cuts:
#   IS_SOFT     for xi < XICUT
#   IS_COLLINEAR for y > 1 - DELTA


def _read_fks_powers(inc_path: str | Path) -> tuple[float, float]:
    """Read deltaO and xicut from a Fortran fks_powers.inc file.

    Returns (delta, xicut) where delta corresponds to deltaO.
    """
    text = Path(inc_path).read_text()

    def _get(name: str) -> float:
        m = re.search(
            rf"parameter\s*\(\s*{name}\s*=\s*([\d.]+)d([+\-]?\d+)\s*\)",
            text,
            re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"'{name}' not found in {inc_path}")
        return float(m.group(1)) * 10 ** int(m.group(2))
    return _get("deltaO"), _get("xicut")


def _covariant2(
    p1: torch.Tensor,
    p2: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    assert p1.shape == p2.shape and p1.shape[-1] == 4
    assert p1.dtype == p2.dtype
    g = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=p1.dtype, device=p1.device)
    return torch.sum(p1 * g * p2, dim=-1, keepdim=keepdim)


def _unit(
    v: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return v / n


def _spa(
    v4: torch.Tensor,
) -> torch.Tensor:
    return v4[..., 1:]


def _spa_norm(
    v4: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.norm(_spa(v4), dim=-1)


def _temp(
    v4: torch.Tensor,
) -> torch.Tensor:
    return v4[..., 0]


def orthonormal_triad_from_axis(
    n_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-15
    n_hat = _unit(n_axis, eps=eps)

    ref_z = torch.tensor(
        [0.0, 0.0, 1.0], dtype=n_axis.dtype, device=n_axis.device
    ).expand_as(n_axis)
    ref_y = torch.tensor(
        [0.0, 1.0, 0.0], dtype=n_axis.dtype, device=n_axis.device
    ).expand_as(n_axis)
    use_alt = (torch.abs((n_hat * ref_z).sum(-1)) > 0.9).unsqueeze(-1)
    ref = torch.where(use_alt, ref_y, ref_z)
    e1 = torch.cross(ref, n_hat, dim=-1)
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(n_hat, e1, dim=-1)
    return n_hat, e1, e2


def _compute_global_kinematics(
    initials_barred_p_ext: torch.Tensor,
    sister_barred_p_ext: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = initials_barred_p_ext.sum(1)
    q0 = _temp(q)
    q2 = _covariant2(q, q)
    k_rec = q - sister_barred_p_ext
    return q, q0, q2, k_rec


def _delta_theta(theta1, theta2) -> torch.Tensor:
    diff = torch.abs(theta1 - theta2)
    # wrap into [0, 2π)
    diff = diff % (2 * torch.pi)
    # reflect anything > π back into [0, π]
    diff = torch.minimum(diff, 2 * torch.pi - diff)
    return diff


def _calculate_beta(q, k_rec):
    numerator = _covariant2(q, q) - (k_rec[:, 0] + _spa_norm(k_rec)) ** 2
    denominator = _covariant2(q, q) + (k_rec[:, 0] + _spa_norm(k_rec)) ** 2
    return (numerator / denominator).clamp(-0.999999999999, 0.999999999999)


def _boost(
    batch_size: int,
    beta: torch.Tensor,
    k_rec: torch.Tensor,
) -> torch.Tensor:
    device = beta.device
    dtype = beta.dtype

    gamma = 1.0 / torch.sqrt(1 - beta**2)
    n = _unit(_spa(k_rec))
    beta_n = beta.unsqueeze(1) * n

    # Prepare boost matrices
    boost = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    boost[:, 0, 0] = gamma
    boost[:, 0, 1:4] = -gamma.unsqueeze(1) * beta_n
    boost[:, 1:4, 0] = -gamma.unsqueeze(1) * beta_n
    gamma_m1 = (gamma - 1.0).unsqueeze(1)
    n_expand = n.unsqueeze(2)
    n_expand_T = n.unsqueeze(1)
    nnT = torch.bmm(n_expand, n_expand_T)
    boost[:, 1:4, 1:4] += gamma_m1[:, None] * nnT
    return boost


def _apply_boost(
    x: torch.Tensor,
    boost_matrices: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(boost_matrices.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)


def compute_dPhi_rad(
    xi: torch.Tensor,
    y: torch.Tensor,
    q2: torch.Tensor,
    P: torch.Tensor,
    sister_barred_p_ext: torch.Tensor,
    rad_jac: torch.Tensor,
    is_soft: bool = False,
    is_coll: bool = False,
) -> torch.Tensor:

    dPhi_rad = (
        2
        * q2
        / ((4 * torch.pi) ** 3)
        * P
        / _spa_norm(sister_barred_p_ext)
        / (2 - xi * (1 - y))
        * rad_jac
    )
    if is_coll:
        return q2 / ((4 * torch.pi) ** 3) * P / _spa_norm(sister_barred_p_ext) * rad_jac
    elif is_soft:
        return q2 / ((4 * torch.pi) ** 3) * rad_jac
    else:
        return dPhi_rad


class FKSInverseConstructor:
    def __init__(self, params, outgoing_pids, verbose=False):
        self.e_cm = params["e_cm"]
        self.verbose = verbose
        self.outgoing_pids = outgoing_pids
        self.n_finals = len(self.outgoing_pids)
        self.xicut = float(params['cross_sections']['nlo'].get('xi_cut', 0.5))
        self.delta = float(params['cross_sections']['nlo'].get('delta_cut', 1.0))

    def _make_emitter_second_to_last(
        self,
        fks: torch.Tensor,
        p_born: torch.Tensor,
        p_real: torch.Tensor,
        p_soft: torch.Tensor,
        p_coll: torch.Tensor,
        debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is a function to reorder particles because MG apis expect
        the emitter to be the second to last particle in the case of 4 final state particles,
        while we currently always put the emitter as the last particle for simplicity in the FKS construction.
        """
        assert (
            p_born.shape[1] == 6
            and p_real.shape[1] == 7
            and p_soft.shape[1] == 7
            and p_coll.shape[1] == 7
        ), "This function is only implemented for the case of 4 final state particles (5 particles in real emission)."
        mask = fks >= 4
        tmp = p_born.clone()
        if debug:
            print("Switching momenta of size (p_born)", p_born.shape)
        p_out_born = p_born.clone()
        p_out_born[mask, 4, :] = tmp[mask, 5, :]
        p_out_born[mask, 5, :] = tmp[mask, 4, :]

        tmp = p_real.clone()
        p_out_real = p_real.clone()
        p_out_real[mask, 4, :] = tmp[mask, 5, :]
        p_out_real[mask, 5, :] = tmp[mask, 6, :]
        p_out_real[mask, 6, :] = tmp[mask, 4, :]

        tmp = p_soft.clone()
        p_out_soft = p_soft.clone()
        p_out_soft[mask, 4, :] = tmp[mask, 5, :]
        p_out_soft[mask, 5, :] = tmp[mask, 6, :]
        p_out_soft[mask, 6, :] = tmp[mask, 4, :]

        tmp = p_coll.clone()
        p_out_coll = p_coll.clone()
        p_out_coll[mask, 4, :] = tmp[mask, 5, :]
        p_out_coll[mask, 5, :] = tmp[mask, 6, :]
        p_out_coll[mask, 6, :] = tmp[mask, 4, :]
        return p_out_born, p_out_real, p_out_soft, p_out_coll

    def radiation_sampler(
        self,
        p_ext: torch.Tensor,
        weights: torch.Tensor,
        sample_size: int,
        rad_us: torch.Tensor = None,
        fks_sector: torch.Tensor = None,
        sampling_strategy: str = "quadratic",
    ) -> torch.Tensor:
        self.p_ext = p_ext
        self.weights = weights
        self.sample_size = sample_size

        self.fks_pos, self.fks_sec = self.sample_fks_sector(self.sample_size, fks_sector)
        sister_barred_p_ext = self.p_ext[torch.arange(self.sample_size), self.fks_pos, :]
        self.q, self.q0, self.q2, self.k_rec = _compute_global_kinematics(
            initials_barred_p_ext=self.p_ext[:, :2, :],
            sister_barred_p_ext=sister_barred_p_ext,
        )
        self.M_rec2 = _covariant2(self.k_rec, self.k_rec)
        self.xi_max = self._calculate_xi_max(self.q2, self.M_rec2, self.q0)

        # Sample u_xi, u_y, u_phi
        u_xi = torch.rand((self.sample_size,)) if rad_us is None else rad_us[:, 0]
        u_y = torch.rand((self.sample_size,)) if rad_us is None else rad_us[:, 1]
        u_phi = torch.rand((self.sample_size,)) if rad_us is None else rad_us[:, 2]
        if sampling_strategy == "uniform":
            xi = self.xi_max * u_xi
            jac_xi = self.xi_max
            y = 1 - 2 * (5e-7 + (1 - 5e-7) * u_y)
            jac_y = 2 * (1 - 5e-7)
            phi = 2.0 * torch.pi * u_phi
            jac_phi = 2.0 * torch.pi
        elif sampling_strategy == "quadratic":
            xi = self.xi_max * u_xi**2
            jac_xi = 2 * self.xi_max * u_xi
            y = 1 - 2 * (5e-7 + (1 - 5e-7) * u_y**2)
            jac_y = 4 * (1 - 5e-7) * u_y
            phi = 2.0 * torch.pi * u_phi
            jac_phi = 2.0 * torch.pi
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        rad_vars = torch.stack([xi, y, phi], dim=-1)
        self.rad_jac = jac_xi * jac_y * jac_phi
        return rad_vars

    def _calculate_xi_max(self, q2, M_rec2, q0):
        return (q2 - M_rec2) / q0**2

    def glu_p3norm(self, xi: torch.Tensor) -> torch.Tensor:
        # |p3| = xi * sqrt(s)/2 = xi * ebeam
        return xi * self.e_cm / 2.0

    def _compute_sister_new3norm(self, q2, M_rec2, q0, glu_p3_norm, y):
        numerator = q2 - M_rec2 - 2 * q0 * glu_p3_norm
        denominator = 2 * (q0 - glu_p3_norm * (1 - y))
        return numerator / denominator

    def sample_fks_sector(
        self, B: int, fks_sec: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fks_sec = (
            torch.randint(1, 7, (B,)) if fks_sec is None else fks_sec + 1
        )  # madnis samples 0-index
        positions = fks_sec.clone() + 1
        positions[fks_sec >= 3] = self.n_finals + 1
        return positions, fks_sec

    def symmetry_factors(self) -> torch.Tensor:
        n_particles = self.p_ext.shape[1]

        fks_symmetry_factors = torch.full((self.sample_size,), 4.0)
        if (n_particles - 2) == 3:
            fks_symmetry_factors[self.fks_sec == 4] = 4
            fks_symmetry_factors[self.fks_sec == 5] = 8
            fks_symmetry_factors[self.fks_sec == 6] = 2
        elif (n_particles - 2) == 4:
            fks_symmetry_factors[self.fks_sec == 1] = 6
            fks_symmetry_factors[self.fks_sec == 2] = 6
            fks_symmetry_factors[self.fks_sec == 3] = 12
            fks_symmetry_factors[self.fks_sec == 4] = 4
            fks_symmetry_factors[self.fks_sec == 5] = 8
            fks_symmetry_factors[self.fks_sec == 6] = 2
        else:
            raise NotImplementedError(
                f"Symmetry factors only implemented for {n_particles} particles."
            )
        return fks_symmetry_factors

    def construct_momenta_from_radiation(
        self, rad_vars: torch.Tensor, weights: torch.Tensor
    ):
        # Unpack radiation vars
        epsilon = 1.0e-10

        xi, y, phi = rad_vars[:, 0], rad_vars[:, 1], rad_vars[:, 2]
        theta = torch.acos(y)

        # This is a block to obtain the normalized k/xi vector for the counterterm
        # Since simply dividing k by xi when it will be needed leads to MadGraph crashing
        sister_barred_p_ext = self.p_ext[torch.arange(self.sample_size), self.fks_pos, :]
        theta_g_kbar_counterterm = torch.atan2(
            sister_barred_p_ext[:, 0] * torch.sin(theta),
            (sister_barred_p_ext[:, 0] * y),
        )
        barred_np, e1, e2 = orthonormal_triad_from_axis(_spa(sister_barred_p_ext))
        K_perp_ct = (
            0.5 * self.q0 * torch.sqrt(1 - torch.cos(theta_g_kbar_counterterm) ** 2)
        )
        u_perp_ct = torch.cos(phi).unsqueeze(-1) * e1 + torch.sin(phi).unsqueeze(-1) * e2
        Kvec_ct = (0.5 * self.q0 * torch.cos(theta_g_kbar_counterterm)).unsqueeze(
            -1
        ) * barred_np + K_perp_ct.unsqueeze(-1) * u_perp_ct
        k_ct = torch.cat([0.5 * self.q0.unsqueeze(-1), Kvec_ct], dim=-1)

        # Compute gluon 3-mom norm and the sister new 3-momentum norm
        K = self.glu_p3norm(xi)
        P = self._compute_sister_new3norm(
            self.q2, _covariant2(self.k_rec, self.k_rec), self.q0, K, y
        )

        # Angle of gluon relative to barred sister (theta is wrt UNbarred sister)
        theta_g_kbar = torch.atan2(P * torch.sin(theta), (P * y + K))
        if not torch.all((theta >= 0) & (theta <= torch.pi)):
            print(
                f"WARNING: Sampled theta values outside [0, pi] range at positions {torch.nonzero((theta < 0) | (theta > torch.pi), as_tuple=False).squeeze(-1).tolist()}"
            )
        theta_k_kbar = _delta_theta(theta, theta_g_kbar)
        torch.clip(
            P, min=torch.zeros_like(P)
        )  # "New 3-momentum norm of n-th particle must be positive"

        K_perp = K * torch.sqrt(1 - torch.cos(theta_g_kbar) ** 2)
        u_perp = torch.cos(phi).unsqueeze(-1) * e1 + torch.sin(phi).unsqueeze(-1) * e2
        Kvec = (K * torch.cos(theta_g_kbar)).unsqueeze(-1) * barred_np + K_perp.unsqueeze(
            -1
        ) * u_perp
        P_perp = P * torch.sqrt(1 - torch.cos(theta_k_kbar) ** 2)
        u_perp = (
            torch.cos(torch.pi + phi).unsqueeze(-1) * e1
            + torch.sin(torch.pi + phi).unsqueeze(-1) * e2
        )
        Pvec = (P * torch.cos(theta_k_kbar)).unsqueeze(-1) * barred_np + P_perp.unsqueeze(
            -1
        ) * u_perp
        close = torch.isclose(_unit(Pvec + Kvec), barred_np).all(dim=-1)
        if not close.all():
            print(
                f"WARNING: Sister and gluon sum are not aligned to original n-th particle direction at positions {torch.nonzero(~close, as_tuple=False).squeeze(-1).tolist()}"
            )

        # Form new 4-vectors for sister and gluon
        p = torch.cat([P.unsqueeze(-1), Pvec], dim=-1)
        k = torch.cat([K.unsqueeze(-1), Kvec], dim=-1)

        new_k_rec = self.q - (p + k)
        if not torch.allclose(
            _covariant2(new_k_rec, new_k_rec),
            _covariant2(self.k_rec, self.k_rec),
            rtol=1e-3,
        ):
            print(
                f"WARNING: Reconstructed recoil mass does not match the barred one at positions {torch.nonzero(~torch.isclose(_covariant2(new_k_rec, new_k_rec), _covariant2(self.k_rec, self.k_rec), rtol=1e-3), as_tuple=False).squeeze(-1).tolist()}"
            )
        k_rec = new_k_rec

        beta = _calculate_beta(self.q, k_rec)
        finals_idx = torch.arange(2, 2 + self.n_finals).expand(self.sample_size, -1)
        spectator_idx = finals_idx[finals_idx != self.fks_pos.unsqueeze(1)].view(
            self.sample_size, self.n_finals - 1
        )
        spectators = self.p_ext.gather(1, spectator_idx.unsqueeze(-1).expand(-1, -1, 4))
        boost_matrix = _boost(spectators.shape[0], beta, k_rec)
        spectators_boosted = _apply_boost(spectators, boost_matrix)
        theta_close = torch.isclose(
            torch.acos(y),
            torch.acos(
                torch.einsum("...i,...i->...", _spa(p), _spa(k))
                / (_spa_norm(p) * _spa_norm(k))
            ),
            atol=1e-10,
        )
        if not theta_close.all():
            print(
                f"WARNING: Angle between sister and gluon is not equal to sampled angle at positions {torch.nonzero(~theta_close, as_tuple=False).squeeze(-1).tolist()}"
            )

        p_ext_regular = self.p_ext.clone()
        p_ext_regular[
            torch.arange(
                self.sample_size,
            ),
            self.fks_pos.long(),
            :,
        ] = p  # modify sister to new momenta
        p_ext_regular.scatter_(
            1, spectator_idx.unsqueeze(-1).expand(-1, -1, 4), spectators_boosted
        )  # Modify spectators to boosted momenta in regular events
        p_ext_regular = torch.cat(
            [p_ext_regular, k.unsqueeze(1)], dim=1
        )  # Regular events

        if self.verbose:
            print("\n----- PS Volume check -----")
        cuts = PhaseSpaceCuts(
            pids=[2, -2, 21] + [21] * (self.n_finals - 3 + 1),
            nparticles=self.n_finals + 1,
            cuts={"pt": {"jets": 20.0}, "eta": {"jets": 5.0}, "dR": {"jj": 0.4}},
        )

        dPhi_rad = compute_dPhi_rad(
            xi=xi,
            y=y,
            q2=self.q2,
            P=P,
            sister_barred_p_ext=sister_barred_p_ext,
            rad_jac=self.rad_jac,
        )
        weights_rad = weights * dPhi_rad
        cut_mask = cuts.cut(p_ext_regular[:, 2:])
        weights_rad[~cut_mask] = 0.0
        if self.verbose:
            print("Number of 0 weights before cutting n+1", (weights == 0).sum().item())
            print(
                "Positions of 0 weights before cutting n+1",
                torch.nonzero(weights == 0, as_tuple=False).squeeze(-1).tolist()[:20],
            )
            print(
                "Number of 0 weights after cutting n+1", (weights_rad == 0).sum().item()
            )
            print(
                "Positions of 0 weights after cutting n+1",
                torch.nonzero(weights_rad == 0, as_tuple=False).squeeze(-1).tolist()[:20],
            )
            print(
                "How many zeroes were added by the n+1 cuts:",
                ((weights_rad == 0) & (weights != 0)).sum().item(),
            )
            print(
                "How many zeroes from n+1 were already zero in n:",
                ((weights_rad == 0) & (weights == 0)).sum().item(),
            )

        rambo_n_plus_one = RamboOnDiet(self.n_finals + 1)
        (p_np1,), weights_n_plus_one = rambo_n_plus_one.map(
            [
                torch.rand(
                    (self.sample_size, 3 * (self.n_finals + 1) - 4),
                ),
                torch.full(
                    (self.sample_size,),
                    self.e_cm,
                ),
            ]
        )
        cut_mask = cuts.cut(p_np1[:, 2:])
        weights_n_plus_one[~cut_mask] = 0.0
        if self.verbose:
            print("\n-------- Without cuts -------")
            print(
                f"Rambo (n={self.n_finals + 1} body)",
                torch.tensor(
                    rambo_n_plus_one._massles_weight(self.e_cm),
                ),
            )
            print(
                f"dPhibar_{self.n_finals} * dPhi_rad:",
                (weights * dPhi_rad * xi).mean(),
                "±",
                (weights * dPhi_rad * xi).std() / (len(weights * dPhi_rad * xi)) ** 0.5,
            )

            print("\n-------- With cuts -------")
            print(
                f"Rambo (n={self.n_finals + 1} body)",
                weights_n_plus_one.mean(),
                "±",
                weights_n_plus_one.std() / (len(weights_n_plus_one)) ** 0.5,
            )
            print(
                f"dPhibar_{self.n_finals} * dPhi_rad:",
                (weights_rad * xi).mean(),
                "±",
                (weights_rad * xi).std() / (len(weights_rad * xi)) ** 0.5,
            )
            print("\n")

        if self.verbose and not torch.isclose(
            (weights_rad * xi).mean(),
            weights_n_plus_one.mean(),
            atol=5
            * (
                (((weights_rad * xi)).std() / (len((weights_rad * xi))) ** 0.5) ** 2
                + (weights_n_plus_one.std() / (len(weights_n_plus_one)) ** 0.5) ** 2
            )
            ** 0.5,
        ):
            print(
                f"Phase space volume does not match expected value within 5 sigmas:\n dPhibar_{self.n_finals} * dPhi_rad: {(weights_rad * xi).mean().item()} ± {((weights_rad * xi)).std().item() / (len(weights_rad * xi))**0.5}\n Rambo (n={self.n_finals + 1} body): {weights_n_plus_one.mean().item()} ± {weights_n_plus_one.std().item() / (len(weights_n_plus_one))**0.5}"
            )
        if self.verbose:
            res = p_ext_regular[:, 2:].sum(dim=1) - self.p_ext[:, :2, :].sum(
                dim=1
            )  # check momentum conservation
            print("\n----- Momenta conservation check -----")
            print(
                "max delta_E, delta_p:",
                float(res[:, 0].abs().max()),
                float(torch.linalg.norm(res[:, 1:], dim=-1).max()),
            )
        epsilon = self.sample_size * (100 * (self.n_finals - 1) + 150)
        momenta_conserved = torch.isclose(
            p_ext_regular[:, 2:].sum(1),
            torch.tensor([self.e_cm, 0.0, 0.0, 0.0]),
            rtol=epsilon * torch.finfo().eps,
        ).any(dim=1)

        if not torch.all(momenta_conserved):
            print(
                f"WARNING: Momentum not conserved after radiation. Results may be incorrect.\nPositions:\n{torch.nonzero(~momenta_conserved, as_tuple=False).squeeze(-1).tolist()}\nElements:\n{p_ext_regular[torch.nonzero(~momenta_conserved, as_tuple=False).squeeze(-1), 2:]}"
            )
        # Collinearity and softness
        IS_COLLINEAR = torch.zeros(
            self.sample_size,
            dtype=torch.bool,
        )
        IS_SOFT = torch.zeros(self.sample_size, dtype=torch.bool)
        IS_SOFT[(xi < self.xicut)] = True
        IS_COLLINEAR[y > 1 - self.delta] = True
        if self.verbose:
            print("\n----- Softness and collinearity -----")
            print(
                "Events neither soft nor collinear:",
                (~(IS_SOFT | IS_COLLINEAR)).sum().item(),
            )
            print("Number of purely soft events:", (IS_SOFT & ~IS_COLLINEAR).sum().item())
            print(
                "Number of purely collinear events:",
                (~IS_SOFT & IS_COLLINEAR).sum().item(),
            )
            print(
                "Number of soft-collinear events:",
                (IS_SOFT & IS_COLLINEAR).sum().item(),
            )

        ### SOFT CASE ###
        soft_mask = IS_SOFT
        xi_soft = torch.zeros_like(xi[soft_mask])
        y_soft = y[soft_mask]

        ### PURE COLLINEAR CASE ###
        coll_mask = IS_COLLINEAR
        xi_coll = xi[coll_mask]
        y_coll = torch.ones_like(y[coll_mask])

        # Form soft radiation events
        P_soft = P[soft_mask].clone()
        P_soft = self._compute_sister_new3norm(
            self.q2[soft_mask],
            self.M_rec2[soft_mask],
            self.q0[soft_mask],
            torch.zeros_like(K[soft_mask]),
            y_soft,
        )
        # Append soft radiation events
        p_ext_soft = torch.cat(
            [self.p_ext[soft_mask], torch.zeros((soft_mask.sum(), 1, 4))], dim=1
        )

        # Form collinear radiation events
        K_col = self.glu_p3norm(xi_coll)
        collinear_dir, _, _ = orthonormal_triad_from_axis(
            _spa(sister_barred_p_ext[coll_mask])
        )
        k_col = torch.cat(
            [K_col.unsqueeze(-1), K_col.unsqueeze(-1) * collinear_dir], dim=-1
        )  # [B_col,4]
        P_coll = self._compute_sister_new3norm(
            self.q2[coll_mask],
            self.M_rec2[coll_mask],
            self.q0[coll_mask],
            K_col,
            y=y_coll,
        )
        p_collinear_sister = torch.cat(
            [P_coll.unsqueeze(-1), P_coll.unsqueeze(-1) * collinear_dir], dim=-1
        )
        p_ext_with_collinear_sister = self.p_ext.clone()
        p_ext_with_collinear_sister[
            torch.nonzero(coll_mask, as_tuple=False).squeeze(),
            self.fks_pos[coll_mask].long(),
            :,
        ] = p_collinear_sister  # Modify sister to new momenta in the collinear case
        p_ext_with_collinear_sister = p_ext_with_collinear_sister[coll_mask]
        # Append collinear radiation events
        p_ext_with_collinear_sister = torch.cat(
            [p_ext_with_collinear_sister, k_col.unsqueeze(1)], dim=1
        )

        # Form final 4-vectors with FKS sectors and soft-collinear indeces
        tmpl = torch.full((self.sample_size, self.p_ext.shape[1] + 1, 4), -1.0)
        p_soft = tmpl.clone()
        p_soft[soft_mask] = p_ext_soft
        p_coll = tmpl.clone()
        p_coll[coll_mask] = p_ext_with_collinear_sister

        # Radiation jacobians for counterterms
        dPhi_soft = torch.zeros_like(dPhi_rad)
        dPhi_soft[soft_mask] = compute_dPhi_rad(
            xi=xi_soft,
            y=y_soft,
            q2=self.q2[soft_mask],
            P=P_soft,
            sister_barred_p_ext=sister_barred_p_ext[soft_mask],
            rad_jac=self.rad_jac[soft_mask],
            is_soft=True,
        )
        dPhi_coll = torch.zeros_like(dPhi_rad)
        dPhi_coll[coll_mask] = compute_dPhi_rad(
            xi=xi_coll,
            y=y_coll,
            q2=self.q2[coll_mask],
            P=P_coll,
            sister_barred_p_ext=sister_barred_p_ext[coll_mask],
            rad_jac=self.rad_jac[coll_mask],
            is_coll=True,
        )

        fks_symmetry_factors = self.symmetry_factors()

        softcoll_correction_factor = torch.ones_like(dPhi_rad)
        softcoll_correction_factor[soft_mask] = 1 / (
            xi[soft_mask] * (1 - y[soft_mask])
        ) + 1 / (1 - y[soft_mask]) / torch.clamp(
            self.xi_max[soft_mask], max=self.xicut
        ) * torch.log(
            self.xicut / (torch.clamp(self.xi_max[soft_mask], max=self.xicut))
        )

        return (
            [p_ext_regular, fks_symmetry_factors * weights * dPhi_rad],
            [
                p_soft,
                fks_symmetry_factors * weights * dPhi_soft * softcoll_correction_factor,
            ],
            [p_coll, fks_symmetry_factors * weights * dPhi_coll],
            k_ct,
            self.fks_sec,
        )
