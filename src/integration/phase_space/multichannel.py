import json
import torch
import madevent7 as me
from madnis.integrator import Integrand, ChannelGrouping
from src.integration.integrand import CrossSection
from src.integration.phase_space.cluster import ClusterJets
from src.integration.phase_space.fks import FKSInverseConstructor

MASSES = {23: 91.188}


WIDTHS = {23: 2.4955}


def clean_pids(pids: list[int]) -> list[int]:
    pids_out = []
    for pid in pids:
        pid = abs(pid)
        if pid == 81:
            pid = 1
        elif pid == 82:
            pid = 11
        pids_out.append(pid)
    return pids_out


def cwnet_preprocessing(mom_in: torch.Tensor) -> torch.Tensor:
    px, py, pz = mom_in[:, :, 1], mom_in[:, :, 2], mom_in[:, :, 3]
    pT2 = px**2 + py**2 + 1e-6
    log_pT = 0.5 * torch.log(pT2)
    phi = torch.arctan2(py, px)
    eta = torch.arctanh(pz / torch.sqrt(pT2 + pz**2))
    return torch.cat((log_pT, phi, eta), dim=1)


class MadnisPhaseSpace:
    def __init__(self, params: dict, is_nlo: bool):
        self.params = params
        self.is_nlo = is_nlo

        with open(params["diagrams_file"]) as f:
            self.meta = json.load(f)[0]
        self.incoming_pids = clean_pids(self.meta["incoming"])
        self.outgoing_pids = clean_pids(self.meta["outgoing"])
        self.incoming_masses = [MASSES.get(pid, 0.0) for pid in self.incoming_pids]
        self.outgoing_masses = [MASSES.get(pid, 0.0) for pid in self.outgoing_pids]
        if params["multichannel"]:
            self.build_multi_channel_mapping()
        else:
            self.build_single_channel_mapping()
        self.build_jet_clusterer(
            ptmin=self.params["cuts"]["pt"].get("jets", 20.0),
            etamax=self.params["cuts"]["eta"].get("jets", 5.0),
            dR=self.params["cuts"]["dR"].get("jj", 0.4),
        )
        self.build_fks_inverse_constructor()
        self.clustered_jets_n = None
        self.clustered_jets_np1 = None

    def build_jet_clusterer(self, ptmin: float, etamax: float, dR: float):
        self.jet_clusterer = ClusterJets(ptmin, etamax, dR)

    def build_fks_inverse_constructor(self):
        self.fks_inverse_constructor = FKSInverseConstructor(
            self.params, self.outgoing_pids, verbose=False
        )

    def build_single_channel_mapping(self):
        modes = {
            "propagator": me.PhaseSpaceMapping.propagator,
            "rambo": me.PhaseSpaceMapping.rambo,
            "chili": me.PhaseSpaceMapping.chili,
        }
        self.channel_weights = None
        self.channel_grouping = None
        self.mappings = [
            me.PhaseSpaceMapping(
                self.incoming_masses + self.outgoing_masses,
                self.params["e_cm"],
                mode=modes[self.params["single_channel_mode"]],
                leptonic=True,
            )
        ]

    def build_multi_channel_mapping(self):
        diagram_count = self.meta["diagram_count"]

        symfact = []
        topologies = []
        permutations = []
        channel_indices = []
        channel_index = 0

        mappings = []
        for channel_id, channel in enumerate(self.meta["channels"]):
            propagators = []
            for i, pid in enumerate(clean_pids(channel["propagators"])):
                mass = MASSES.get(pid, 0.0)
                width = WIDTHS.get(pid, 0.0)
                propagators.append(me.Propagator(mass=mass, width=width))
            vertices = channel["vertices"]
            diagrams = channel["diagrams"]
            chan_permutations = [d["permutation"] for d in diagrams]
            diag = me.Diagram(
                self.incoming_masses, self.outgoing_masses, propagators, vertices
            )
            chan_topology = me.Topology(diag)
            channel_index_first = channel_index
            symfact_index_first = len(symfact)
            channel_index += 1
            symfact.append(None)
            for d in diagrams[1:]:
                channel_index += 1
                symfact.append(symfact_index_first)

            topologies.append(chan_topology)
            permutations.append(chan_permutations)
            channel_indices.append(list(range(channel_index_first, channel_index)))
            mappings.append(
                me.PhaseSpaceMapping(
                    chan_topology,
                    self.params["e_cm"],
                    invariant_power=0.7,
                    permutations=chan_permutations,
                    leptonic=True,
                )
            )

        self.channel_weights = me.PropagatorChannelWeights(
            topologies, permutations, channel_indices
        )
        self.channel_grouping = ChannelGrouping(symfact)
        self.channel_id_map = torch.tensor(
            [channel.group.group_index for channel in self.channel_grouping.channels],
        )
        self.group_indices = torch.tensor(
            [channel.position_in_group for channel in self.channel_grouping.channels],
            dtype=torch.int32,
        )
        self.mappings = mappings

    def call_mapping_and_channel_weights(
        self, r: torch.Tensor, channel: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if len(self.mappings) == 1:
            (p_ext, x1, x2), jac = self.mappings[0].map_forward([r])
            return p_ext, jac
        else:
            n = len(r)
            p_ext = r.new_zeros((n, len(self.outgoing_pids) + 2, 4))
            jac = r.new_zeros((n,))
            integration_channel = self.channel_id_map[channel]
            index_in_group = self.group_indices[channel]
            for i, mapping in enumerate(self.mappings):
                chan_mask = integration_channel == i
                (p_ext[chan_mask], _, _), jac[chan_mask] = mapping.map_forward(
                    [r[chan_mask]],
                    [] if mapping.channel_count() <= 1 else [index_in_group[chan_mask]],
                )
            cwnet_input = cwnet_preprocessing(p_ext[:, 2:])
            channel_weights = self.channel_weights(p_ext)
            return p_ext, jac, cwnet_input, channel_weights

    def multichannel_args(self):
        if len(self.mappings) == 1:
            return {}
        else:
            return dict(
                channel_count=len(self.channel_grouping.channels),
                remapped_dim=3 * len(self.outgoing_pids),
                has_channel_weight_prior=True,
                channel_grouping=self.channel_grouping,
            )

    def cluster_jets(self, p_ext: torch.Tensor) -> torch.Tensor:
        clustered_jets, njets = self.jet_clusterer.cluster(p_ext[:, 2:])
        return clustered_jets, njets

    def cluster_and_cut(
        self, p_n: torch.Tensor, cut_mask: torch.Tensor, min_njets: int
    ) -> torch.Tensor:
        clustered_jets, njets = self.cluster_jets(p_n)
        # Demand at least n separated jets from the n final particles
        cut_mask = cut_mask & (njets >= min_njets)
        return cut_mask, clustered_jets

    def _lo_ps_from_r(self, r: torch.Tensor, channel: torch.Tensor | None):

        p_born, weights_born, *rest = self.call_mapping_and_channel_weights(r, channel)
        cut_mask = weights_born != 0
        cut_mask, self.clustered_jets_n = self.cluster_and_cut(
            p_born, cut_mask, p_born.shape[1] - 2
        )
        weights_born[~cut_mask] = 0.0
        return p_born, weights_born, rest

    def _nlo_ps_from_r(
        self,
        r: torch.Tensor,
        channel: torch.Tensor | None = None,
        prev_clustered_jets_n: torch.Tensor | None = None,
        prev_clustered_jets_np1: torch.Tensor | None = None,
    ):
        fks_sector = r[:, 0].long()
        r_born = r[:, 1:-3]
        r_rad = r[:, -3:]
        p_born, weights_born, *rest = self.call_mapping_and_channel_weights(
            r_born, channel
        )
        cut_mask_born = weights_born != 0

        # This maps radiation random numbers to the radiation variables, which are dependent on p_born
        rad_vars = self.fks_inverse_constructor.radiation_sampler(
            p_ext=p_born,
            weights=weights_born,
            sample_size=weights_born.shape[0],
            rad_us=r_rad,
            fks_sector=fks_sector,
            sampling_strategy="quadratic",
        )
        # Construct n+1 phase space needed for NLO evaluation
        (
            real,
            soft,
            coll,
            k_ct,
            fks,
        ) = self.fks_inverse_constructor.construct_momenta_from_radiation(
            rad_vars, weights=weights_born
        )
        p_real, weights_real = real
        p_soft, weights_soft = soft
        p_coll, weights_coll = coll
        cut_mask_real = weights_real != 0
        if p_born.shape[1] - 2 == 4:
            # MG API for epem -> uuxgg expects the emitter to be the
            # second-to-last particle in FKS sectors 4,5,6
            (
                p_born,
                p_real,
                p_soft,
                p_coll,
            ) = self.fks_inverse_constructor._make_emitter_second_to_last(
                fks, p_born, p_real, p_soft, p_coll, debug=False
            )

        # Apply cuts and masks
        cut_mask_born, clustered_jets_n = self.cluster_and_cut(
            p_born, cut_mask_born, p_born.shape[1] - 2
        )
        cut_mask_real, clustered_jets_np1 = self.cluster_and_cut(
            p_real, cut_mask_real, p_born.shape[1] - 2
        )
        if prev_clustered_jets_n is not None:
            self.clustered_jets_n = prev_clustered_jets_n + clustered_jets_n
        else:
            self.clustered_jets_n = clustered_jets_n
        if prev_clustered_jets_np1 is not None:
            self.clustered_jets_np1 = prev_clustered_jets_np1 + clustered_jets_np1
        else:
            self.clustered_jets_np1 = clustered_jets_np1

        weights_born[~cut_mask_born] = 0.0
        weights_real[~cut_mask_real] = 0.0
        weights_soft[~cut_mask_born] = 0.0
        weights_coll[~cut_mask_born] = 0.0

        # weights born needs to be x2 to account for the two flavors of the underlying born uuxg and ccxg
        weights_born = 2.0 * weights_born

        return (
            [
                rad_vars,  # (2,B, 3)
                [p_born, weights_born],  # n-body PS + weights
                [p_real, weights_real],  # (n+1)-body real
                [
                    p_soft,
                    weights_soft,
                ],  # soft counterterm (integrated subtracted for some)
                [p_coll, weights_coll],  # collinear counterterm
                k_ct,  # integrated soft CT (B, 4)
                fks,  # FKS sector info
            ],
            rest,  # (cwnet_input, channel_weights) if multichannel
        )

    def build_lo_integrand(self, cross_section: CrossSection) -> Integrand:
        def func(r: torch.Tensor, channel: torch.Tensor | None = None):
            p_ext, weights, rest = self._lo_ps_from_r(r, channel)
            cut_mask = weights != 0
            weights[cut_mask] *= cross_section(p_ext[cut_mask])
            return (weights, *rest) if len(rest) > 0 else weights

        return Integrand(
            func, input_dim=3 * len(self.outgoing_pids) - 4, **self.multichannel_args()
        )

    def build_nlo_integrand(self, cross_section: CrossSection) -> Integrand:
        def func(r: torch.Tensor, channel: torch.Tensor | None = None):
            nlo_inputs, rest = self._nlo_ps_from_r(r, channel)
            weights = cross_section(nlo_inputs)
            return (weights, *rest) if len(rest) > 0 else weights

        return Integrand(
            func,
            input_dim=3 * len(self.outgoing_pids)
            - 4
            + 3
            + 1,  # extra particle (+3) and fks sector (+1) for NLO
            discrete_dims=[6],
            discrete_dims_position="first",
            **self.multichannel_args()
        )

    def build_integrand(self, cross_section: CrossSection) -> Integrand:
        if self.is_nlo:
            return self.build_nlo_integrand(cross_section)
        else:
            return self.build_lo_integrand(cross_section)
