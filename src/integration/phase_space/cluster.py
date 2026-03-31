import torch
import fastjet as fj


class ClusterJets:
    """
    Cluster per-event inclusive jets from batched 4-vectors.

    Input shape: (B, Npart, 4) with (E, px, py, pz) as a torch.Tensor.
    Returns per-event jet tensors (variable-length) and counts above ptmin.
    """

    def __init__(
        self,
        ptmin=20.0,
        etamax=5.0,
        dR=0.4,
        algo=fj.kt_algorithm,
    ):
        self.jetdef = fj.JetDefinition(algo, dR)
        self.ptmin = float(ptmin)
        self.etamax = float(etamax)

    def cluster(self, p_ext: torch.Tensor):
        """
        p_ext: (B, N, 4)

        Returns:
          jets_per_event: list[torch.Tensor] with shape (njets[i], 4)
          njets:          torch.Tensor of shape (B,) with counts
        """
        assert p_ext.dim() == 3 and p_ext.size(-1) == 4, "p_ext must be (B, N, 4)"

        device = p_ext.device
        dtype = p_ext.dtype
        B, N, _ = p_ext.shape

        p_ext = p_ext.clone().detach().cpu()
        jets_per_event = []
        njets = torch.zeros(B, dtype=torch.long, device=device)
        sel_eta = fj.SelectorAbsEtaMax(self.etamax)
        for idx, event_p4 in enumerate(p_ext):
            # build pseudojets, that take different ordering in FastJet
            pseudojets = [
                fj.PseudoJet(px, py, pz, E) for (E, px, py, pz) in event_p4.tolist()
            ]
            cs = fj.ClusterSequence(pseudojets, self.jetdef)
            jets = sel_eta(cs.inclusive_jets(self.ptmin))
            njets[idx] = len(jets)
            if jets:
                jets_p4 = torch.tensor(
                    [[j.E(), j.px(), j.py(), j.pz()] for j in jets],
                    dtype=dtype,
                )
            else:
                jets_p4 = torch.empty((0, 4), dtype=dtype)
            jets_per_event.append(jets_p4.to(device))
        return jets_per_event, njets
