from utils.plots_utils import *
from src.utils.physics import (
    list_EPxPyPz_to_PtPhiEtaM,
    sort_by_pt,
    delta_phi,
    delta_eta,
    delta_R,
    PtPhiEtaM_to_EPxPyPz,
)

plt.rc("font", family="serif", size=12)
plt.rc("font", serif=["XCharter", "DejaVu Serif"])
mpl.rcParams["text.usetex"] = False

TRUTH_COLOR = "#07078A"
NEUTRAL_COLOR = "black"
NN_COLOR_red = "#8A0707"
NN_COLOR_green = "#2C7D55"
NN_COLOR_purple = "#790679"
NN_COLOR_orange = "darkorange"

bins = {
    3: {
        "pt": np.linspace(0, 500, 31),
        "pt1": np.linspace(0, 500, 31),
        "pt2": np.linspace(0, 500, 31),
        "pt3": np.linspace(0, 400, 31),
        "pt4": np.linspace(0, 300, 31),
        "phi": np.linspace(-np.pi, np.pi, 31),
        "eta": np.linspace(-4, 4, 31),
        "ST": np.linspace(0, 1000, 31),
        "dphi": np.linspace(-np.pi, np.pi, 31),
        "deta": np.linspace(-7, 7, 31),
        "dR": np.linspace(0.4, 7, 31),
        "mjj": np.linspace(0, 1000, 31),
    },
    4: {
        "pt": np.linspace(0, 500, 31),
        "pt1": np.linspace(0, 500, 31),
        "pt2": np.linspace(0, 500, 31),
        "pt3": np.linspace(0, 400, 31),
        "pt4": np.linspace(0, 300, 31),
        "pt5": np.linspace(0, 200, 31),
        "phi": np.linspace(-np.pi, np.pi, 31),
        "eta": np.linspace(-4, 4, 31),
        "ST": np.linspace(0, 1000, 31),
        "dphi": np.linspace(-np.pi, np.pi, 31),
        "deta": np.linspace(-7, 7, 31),
        "dR": np.linspace(0.4, 7, 31),
        "mjj": np.linspace(0, 1000, 31),
    },
}


@dataclass
class Metric:
    name: str
    value: float
    format: str | None = None
    unit: str | None = None
    tex_label: str | None = None


def plot_train_metrics(
    file: str,
    train_results: Optional[Sequence[dict[str, Any]]] = None,
    vegas_train_results: Optional[Sequence[dict[str, Any]]] = None,
    data_file: Optional[str] = None,
    rect: tuple[float, float, float, float] = (0.17, 0.12, 0.96, 0.96),
    logy_losses: bool = False,
):
    """
    Plots training losses and learning rates for VEGAS and MadNIS.
    """
    if data_file is not None:
        payload = {
            "vegas_train_results": list(vegas_train_results)
            if vegas_train_results
            else None,
            "train_results": list(train_results) if train_results else None,
        }
        with open(data_file, "wb") as f:
            pickle.dump(payload, f)

    has_vegas = bool(vegas_train_results)
    has_madnis = bool(train_results)

    with PdfPages(file) as pp:
        # Losses
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if not has_vegas and not has_madnis:
            print("No losses to plot.")
            plt.close(fig)
            return
        vegas_end = 0
        madnis_batch0 = None
        ## VEGAS
        if has_vegas:
            vx = [int(r["batch"]) for r in vegas_train_results]
            vy = [float(r["loss"]) for r in vegas_train_results]
            if vx:
                ax.plot(vx, vy, label="VEGAS", color="C0")
                vegas_end = max(vx)
        ## MadNIS
        lr_x, lr_y = None, None
        if has_madnis:
            mx_raw = [int(r["batch"]) for r in train_results]
            my = [float(r["online_loss"]) for r in train_results]
            if mx_raw:
                madnis_batch0 = mx_raw[0]
            if has_vegas and vegas_end > 0 and madnis_batch0 is not None:
                # Here I start the MadNIS plot exactly at the end of VEGAS pretraining + 1
                mx = [vegas_end + 1 + (b - madnis_batch0) for b in mx_raw]
            else:
                mx = mx_raw

            ax.plot(mx, my, label="MadNIS", color="C3")

            lr_vals = [r.get("learning_rate", None) for r in train_results]
            if any(v is not None for v in lr_vals):
                lr_x = [x for x, v in zip(mx, lr_vals) if v is not None]
                lr_y = [float(v) for v in lr_vals if v is not None]

        if logy_losses:
            ax.set_yscale("log")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_xlim(left=0)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.legend(frameon=False, loc="upper right")
        fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
        fig.align_ylabels(ax)
        fig.savefig(pp, format="pdf")
        plt.close(fig)

        # Learning Rate
        if has_madnis and lr_x is not None and lr_y is not None and len(lr_x) > 0:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(lr_x, lr_y, color="crimson", label="Learning Rate")
            ax.set_xlim(left=0)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.get_major_formatter().set_scientific(False)

            ax.set_xlabel("Iterations")
            ax.set_ylabel("Learning rate")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="upper right")

            fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
            fig.align_ylabels(ax)
            fig.savefig(pp, format="pdf")
            plt.close(fig)


def plot_fks_sectors(
    file: str, fks_sectors: np.ndarray, lines_file: Optional[str] = None, weights: np.ndarray = None
) -> None:
    bins = np.linspace(1, 7, 7) - 0.5
    y, y_err = compute_hist_data_simple_histogram(data=fks_sectors, bins=bins, weights=weights)
    line = [
        Line(
            y=y,
            y_err=y_err,
            label="MadNIS",
            color=NN_COLOR_red,
        )
    ]
    pickle_data = {
        f"fks_lines": line,
        f"fks_bins": bins,
    }
    if lines_file is not None:
        append_to_pickle(pickle_data, lines_file)
    with PdfPages(file) as pp:
        hist_plot(
            pp,
            line,
            bins,
            show_ratios=False,
            fks_hist=True,
            tex_label=r"\mathrm{FKS\ sector\ }(i,j)",
            unit=None,
            no_scale=True,
            ylim=None,
            yscale="linear",
        )


class Plots:
    def __init__(
        self,
        jets_n: list[np.ndarray],
        jets_np1: list[np.ndarray],
        weights_dict: dict[str, np.ndarray],
        n_particles: int = 3,
        surrogates_to_use: str = "none",  # to choose in ["none", "virtual", "virtual_and_real"]
        model_name: Optional[str] = None,
        debug: bool = True,
        sorted_jets_n: Optional[list] = None,
        sorted_jets_np1: Optional[list] = None,
    ):
        self.model_name = (
            (
                "MadNIS"
                if surrogates_to_use == "none"
                else r"MadNIS $+$ (V/B)$_\text{surr}$"
                if surrogates_to_use == "virtual"
                else r"MadNIS + (V/B)$_\text{surr} + \Sigma_\text{surr}$"
            )
            if model_name is None
            else model_name
        )
        self.surrogates_to_use = surrogates_to_use
        self.n_events = len(jets_n)
        if debug:
            print("DEBUG: Plotting for", self.n_events, "events")

        self.n_particles = n_particles
        self.bins = bins[self.n_particles]
        self.process_name = {
            3: "$e^{+} e^{-} \\to u \\bar{u} g$",
            4: "$e^{+} e^{-} \\to u \\bar{u} g g$",
        }[n_particles]
        self.weights_dict = weights_dict.copy()
        self.debug = debug

        # Convert jets to PtPhiEtaM and sort by pt
        if sorted_jets_n is None:
            jets_n_PtPhiEtaM = list_EPxPyPz_to_PtPhiEtaM(jets_n)
            self.jets_n_PtPhiEtaM = sort_by_pt(jets_n_PtPhiEtaM)
        else:
            self.jets_n_PtPhiEtaM = sorted_jets_n
        if sorted_jets_np1 is None:
            jets_np1_PtPhiEtaM = list_EPxPyPz_to_PtPhiEtaM(jets_np1)
            self.jets_np1_PtPhiEtaM = sort_by_pt(jets_np1_PtPhiEtaM)
        else:
            self.jets_np1_PtPhiEtaM = sorted_jets_np1
        if debug:
            Ibefore = (
                self._total_weight_n() + self._total_weight_np1()
            ).sum() / self.n_events
            uIbefore = (
                self._total_weight_n() + self._total_weight_np1()
            ).std() / np.sqrt(self.n_events)
            print(
                f"DEBUG: Integral before normalization: {Ibefore:.5f} +/- {uIbefore:.5f}"
            )
        for k in [l for l in self.weights_dict if "weight" in l and "fks_sec" not in l]:
            self.weights_dict[k] = np.asarray(self.weights_dict[k]) / self.n_events
            if debug:
                print(f"DEBUG: Normalizing weight {k} by {self.n_events}")
            # Important: normalize weights by number of events so that sum(w) gives cross section
        if debug:
            Iafter = (self._total_weight_n() + self._total_weight_np1()).sum()
            uIafter = (self._total_weight_n() + self._total_weight_np1()).std() * np.sqrt(
                self.n_events
            )
            print(f"DEBUG: Integral after normalization: {Iafter:.5f} +/- {uIafter:.5f}")

            for k in range(len(self.jets_n_PtPhiEtaM)):
                if len(self.jets_n_PtPhiEtaM[k]) == 0:
                    print("DEBUG: Position", k, "in n events has no jets:")
                    print(self.jets_n_PtPhiEtaM[k])
            for k in range(len(self.jets_np1_PtPhiEtaM)):
                if len(self.jets_np1_PtPhiEtaM[k]) == 0:
                    print("DEBUG: Position", k, "in np1 events has no jets:")
                    print(self.jets_np1_PtPhiEtaM[k])

    def plot_jet_pt(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        indices = np.arange(self.n_particles + 1)
        labels = [f"j_{i+1}" for i in indices]
        tex_labels = [rf"p_{{T, j_{i+1}}}" for i in indices]
        lines = []
        pickle_data = []
        bins = []
        for jet_idx, jet_label in zip(indices, labels):
            bin = (
                self.bins[f"pt{jet_idx+1}"]
                if f"pt{jet_idx+1}" in self.bins
                else self.bins["pt"]
            )
            l, p = self._get_feature_lines(
                jet_idx, jet_label, bins=bin, feature="pt", sample_file=sample_file
            )
            bins.append(bin)
            lines.append(l)
            pickle_data.append(p)
        # Plotting
        with PdfPages(file) as pp:
            for bin, line, tex_label in zip(bins, lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bin,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit="GeV",
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_phi(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["phi"]
        indices = np.arange(self.n_particles + 1)
        labels = [f"j_{i+1}" for i in indices]
        tex_labels = [rf"\phi_{{j_{{{i+1}}}}}" for i in indices]
        lines = []
        pickle_data = []
        for jet_idx, jet_label in zip(indices, labels):
            l, p = self._get_feature_lines(
                jet_idx, jet_label, bins=bins, feature="phi", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        # Plotting
        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit=None,
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_eta(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["eta"]
        indices = np.arange(self.n_particles + 1)
        labels = [f"j_{i+1}" for i in indices]
        tex_labels = [rf"\eta_{{j_{{{i+1}}}}}" for i in indices]

        lines = []
        pickle_data = []
        for jet_idx, jet_label in zip(indices, labels):
            l, p = self._get_feature_lines(
                jet_idx, jet_label, bins=bins, feature="eta", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit=None,
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )

        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_ST(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["ST"]
        indices = np.arange(self.n_particles + 1)
        pairs = [(i, j) for i in indices for j in indices if i < j]
        labels = [f"j_{i+1}j_{j+1}" for (i, j) in pairs]
        tex_labels = [rf"S_{{T, j_{i+1}j_{j+1}}}" for (i, j) in pairs]
        lines = []
        pickle_data = []
        for (i, j), label in zip(pairs, labels):
            l, p = self._get_delta_lines(
                i, j, label, bins=bins, correlation="ST", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit="GeV",
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_dphi(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["dphi"]
        indices = np.arange(self.n_particles + 1)
        pairs = [(i, j) for i in indices for j in indices if i < j]
        labels = [f"j_{i+1}j_{j+1}" for (i, j) in pairs]
        tex_labels = [rf"\Delta \phi_{{j_{i+1}j_{j+1}}}" for (i, j) in pairs]
        lines = []
        pickle_data = []
        for (i, j), label in zip(pairs, labels):
            l, p = self._get_delta_lines(
                i, j, label, bins=bins, correlation="dphi", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit=None,
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_deta(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["deta"]
        indices = np.arange(self.n_particles + 1)
        pairs = [(i, j) for i in indices for j in indices if i < j]
        labels = [f"j_{i+1}j_{j+1}" for (i, j) in pairs]
        tex_labels = [rf"\Delta \eta_{{j_{i+1}j_{j+1}}}" for (i, j) in pairs]
        lines = []
        pickle_data = []
        for (i, j), label in zip(pairs, labels):
            l, p = self._get_delta_lines(
                i, j, label, bins=bins, correlation="deta", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit=None,
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_dR(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["dR"]
        indices = np.arange(self.n_particles + 1)
        pairs = [(i, j) for i in indices for j in indices if i < j]
        labels = [f"j_{i+1}j_{j+1}" for (i, j) in pairs]
        tex_labels = [rf"\Delta R_{{j_{i+1}j_{j+1}}}" for (i, j) in pairs]
        lines = []
        pickle_data = []
        for (i, j), label in zip(pairs, labels):
            l, p = self._get_delta_lines(
                i, j, label, bins=bins, correlation="dR", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)
        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit=None,
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )
        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def plot_jet_mjj(
        self,
        file: str,
        lines_file: Optional[str] = None,
        sample_file: Optional[str] = None,
    ):
        bins = self.bins["mjj"]  # <-- add this to your bins dict

        indices = np.arange(self.n_particles + 1)
        pairs = [(i, j) for i in indices for j in indices if i < j]

        labels = [f"j_{i+1}j_{j+1}" for (i, j) in pairs]
        tex_labels = [rf"m_{{{i+1}{j+1}}}" for (i, j) in pairs]  # or m(j_i,j_j)

        lines = []
        pickle_data = []
        for (i, j), label in zip(pairs, labels):
            l, p = self._get_delta_lines(
                i, j, label, bins=bins, correlation="mjj", sample_file=sample_file
            )
            lines.append(l)
            pickle_data.append(p)

        with PdfPages(file) as pp:
            for line, tex_label in zip(lines, tex_labels):
                for yscale in ["linear"]:
                    hist_plot(
                        pp,
                        line,
                        bins,
                        show_ratios=False,
                        title=self.process_name
                        if self.process_name is not None
                        else None,
                        tex_label=tex_label,
                        unit="GeV",
                        no_scale=True,
                        ylim=None,
                        yscale=yscale,
                    )

        if lines_file is not None:
            append_to_pickle(pickle_data, lines_file)

    def _get_feature_lines(
        self,
        jet_idx: int,
        jet_label: str,
        bins: list[float],
        feature: str = "pt",
        sample_file: Optional[str] = None,
    ):
        idx = {"pt": 0, "phi": 1, "eta": 2}[feature]
        jet_n_features, valid_n = self._jet_feature_and_mask(
            self.jets_n_PtPhiEtaM, jet_idx, idx
        )
        jet_np1_features, valid_np1 = self._jet_feature_and_mask(
            self.jets_np1_PtPhiEtaM, jet_idx, idx
        )
        if sample_file is not None:
            append_to_pickle(
                {
                    f"{jet_label}_{feature}_n": jet_n_features,
                    f"{jet_label}_{feature}_np1": jet_np1_features,
                    f"{jet_label}_{feature}_weights_n": self._total_weight_n(),
                    f"{jet_label}_{feature}_weights_np1": self._total_weight_np1(),
                    f"{jet_label}_{feature}_valid_n": valid_n,
                    f"{jet_label}_{feature}_valid_np1": valid_np1,
                },
                sample_file,
            )
        y, y_err = compute_hist_data(
            bins=bins,
            data_n=jet_n_features,
            data_np1=jet_np1_features,
            weights_n=self._total_weight_n(),
            weights_np1=self._total_weight_np1(),
            valid_n=valid_n,
            valid_np1=valid_np1,
            bayesian=False,
            debug=self.debug,
        )
        line = [
            Line(
                y=y,
                y_err=y_err,
                label=self.model_name,
                color=NN_COLOR_red,
            )
        ]
        pickle_data = {
            f"{jet_label}_{feature}_lines": line,
            f"{jet_label}_{feature}_bins": bins,
        }
        return line, pickle_data

    def _jet_feature_and_mask(self, jets_list, jet_idx: int, feature_idx: int):
        feature = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)

        for evt, jets in enumerate(jets_list):
            if len(jets) > jet_idx:
                feature[evt] = jets[jet_idx, feature_idx]
                valid[evt] = True

        return feature, valid

    def _get_delta_lines(
        self,
        i: int,
        j: int,
        pair_label: str,
        bins: np.ndarray,
        correlation: str = "dR",  # dphi, deta, dR, ST, mjj
        sample_file: Optional[str] = None,
    ):
        delta_function = {
            "dR": self._deltaR_and_mask,
            "dphi": self._deltaPhi_and_mask,
            "deta": self._deltaEta_and_mask,
            "ST": self._ST_and_mask,
            "mjj": self._mij_and_mask,
        }[correlation]

        delta_n, valid_n = delta_function(self.jets_n_PtPhiEtaM, i, j)
        delta_np1, valid_np1 = delta_function(self.jets_np1_PtPhiEtaM, i, j)

        if sample_file is not None:
            append_to_pickle(
                {
                    f"{pair_label}_{correlation}_n": delta_n,
                    f"{pair_label}_{correlation}_np1": delta_np1,
                    f"{pair_label}_{correlation}_weights_n": self._total_weight_n(),
                    f"{pair_label}_{correlation}_weights_np1": self._total_weight_np1(),
                    f"{pair_label}_{correlation}_valid_n": valid_n,
                    f"{pair_label}_{correlation}_valid_np1": valid_np1,
                },
                sample_file,
            )

        y, y_err = compute_hist_data(
            bins=bins,
            data_n=delta_n,
            data_np1=delta_np1,
            weights_n=self._total_weight_n(),
            weights_np1=self._total_weight_np1(),
            valid_n=valid_n,
            valid_np1=valid_np1,
            bayesian=False,
            debug=self.debug,
        )
        line = [
            Line(
                y=y,
                y_err=y_err,
                label=self.model_name,
                color=NN_COLOR_red,
            )
        ]
        pickle_data = {
            f"{pair_label}_{correlation}_lines": line,
            f"{pair_label}_{correlation}_bins": bins,
        }
        return line, pickle_data

    def _ST_and_mask(self, jets_list, i: int, j: int):
        ST = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)
        for evt, jets in enumerate(jets_list):
            if len(jets) > max(i, j):
                ST[evt] = jets[i, 0] + jets[j, 0]  # sum of pT
                valid[evt] = True
        return ST, valid

    def _deltaPhi_and_mask(self, jets_list, i: int, j: int):
        dphi = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)
        for evt, jets in enumerate(jets_list):
            if len(jets) > max(i, j):
                dphi[evt] = delta_phi(
                    jets[i, 1],
                    jets[j, 1],
                    abs=False,
                )
                valid[evt] = True
        return dphi, valid

    def _deltaEta_and_mask(self, jets_list, i: int, j: int):
        deta = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)
        for evt, jets in enumerate(jets_list):
            if len(jets) > max(i, j):
                deta[evt] = delta_eta(
                    jets[i, 2],
                    jets[j, 2],
                    abs=False,
                )
                valid[evt] = True
        return deta, valid

    def _deltaR_and_mask(self, jets_list, i: int, j: int):
        dr = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)

        for evt, jets in enumerate(jets_list):
            if len(jets) > max(i, j):
                dr[evt] = delta_R(
                    jets[i, 1],
                    jets[i, 2],
                    jets[j, 1],
                    jets[j, 2],
                )
                valid[evt] = True

        return dr, valid

    def _mij_and_mask(self, jets_list, i: int, j: int):
        mij = np.empty(self.n_events, dtype=float)
        valid = np.zeros(self.n_events, dtype=bool)

        for evt, jets in enumerate(jets_list):
            if len(jets) > max(i, j):
                # convert only the two jets we need
                EPxPyPz = PtPhiEtaM_to_EPxPyPz(jets[[i, j]])

                E = EPxPyPz[:, 0].sum()
                px = EPxPyPz[:, 1].sum()
                py = EPxPyPz[:, 2].sum()
                pz = EPxPyPz[:, 3].sum()

                m2 = E**2 - (px**2 + py**2 + pz**2)
                if m2 >= 1e-17:
                    mij[evt] = np.sqrt(max(m2, 0.0))  # numerical safety
                    valid[evt] = True
                else:
                    mij[evt] = 0.0
                    valid[evt] = False
        return mij, valid

    def _total_weight_n(self):
        suffix = ""
        if "virtual" in self.surrogates_to_use:
            suffix = "_V_surrogate"
        if "real" in self.surrogates_to_use:
            suffix = "_VR_surrogate"
        return self.weights_dict[f"nbody_weight{suffix}"]

    def _total_weight_np1(self):
        if "real" in self.surrogates_to_use:
            return self.weights_dict[f"real_weight_surrogate"]
        else:
            return self.weights_dict[f"real_weight"]
