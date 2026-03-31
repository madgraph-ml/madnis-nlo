import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

plt.rc("font", family="serif", size=22)
plt.rc("axes", titlesize="medium")

colors = ["#0000cc", "#b40000"]


class Plots:
    def __init__(
        self,
        preds_mean,
        truth,
        sigma2s,
        doc,
        params,
        label="test",
    ):
        self.doc = doc
        self.params = params
        self.label = label
        self.true_amps = truth
        self.pred_mean_amps = preds_mean
        self.sigmas = np.sqrt(sigma2s)
        self.n_rep = self.params.get("plot_n_reps", 50)

    def plot_amplitudes(
        self,
    ):
        # bins = np.logspace(-9, -3, 21)
        bins = np.linspace(-1, 1, 21)
        values, true_test_bin = [], []
        pred_reps = []
        for _ in range(self.n_rep):
            error = np.random.normal(
                loc=0.0, scale=self.sigmas, size=self.pred_mean_amps.shape
            )
            data_rep = error + self.pred_mean_amps
            pred_reps.append(data_rep)
        pred_reps = np.stack(pred_reps, 0)
        print("shape of data w/ replicas: ", pred_reps.shape)

        # get hist values
        for i in range(len(pred_reps)):
            hist, bin_edges = np.histogram(pred_reps[i], bins=bins)
            # normalize by the number of amps
            hist = hist.astype("float") / hist.sum()
            values.append(hist)
        bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
        values = np.stack(values, axis=1)
        mean_hist = np.mean(values, axis=1)
        std_hist = np.std(values, axis=1)
        true_hist, bin_edges = np.histogram(self.true_amps, bins=bins)
        true_hist = true_hist.astype("float") / true_hist.sum()
        pred_ens_hist, bin_edges = np.histogram(self.pred_mean_amps, bins=bins)
        pred_ens_hist = pred_ens_hist.astype("float") / pred_ens_hist.sum()

        delta_amp = (self.pred_mean_amps / (self.true_amps + 1e-20)) - 1
        pull = (self.pred_mean_amps - self.true_amps) / (self.sigmas + 1e-20)

        with PdfPages(self.doc.basedir + f"/hist_{self.label}_amps.pdf") as pdf:

            fig, ax = plt.subplots(
                3,
                1,
                figsize=[7, 7],
                sharex=True,
                gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.0},
            )
            # add ratio plot for test/true
            ax[0].step(bin_edges[:-1], pred_ens_hist, where="post", label="pred")
            ax[0].step(bin_edges[:-1], true_hist, where="post", label="true", color="C3")
            ax[0].step(bin_edges[:-1], mean_hist, where="post", label="mean", color="C1")
            ax[0].fill_between(
                bin_edges[:-1],
                mean_hist + std_hist,
                mean_hist - std_hist,
                step="post",
                alpha=0.3,
                color="C1",
            )

            ax[0].set_yscale("log")
            # ax[0].set_xscale("log")
            # ax1.set_xlabel( "truth amplitudes" )
            ax[0].set_ylabel("normalized counts")
            ax[0].legend()

            ax[1].set_ylabel(r"$\frac{\text{Model}}{\text{Truth}}$")
            ax[1].set_ylim(0.7, 1.3)
            ax[1].get_xticklabels()

            ratio = mean_hist / (true_hist + 1e-20)
            ratio_isnan = np.isnan(ratio)
            ratio[ratio_isnan] = 1.0
            delta = np.fabs(ratio - 1) * 100

            ax[1].fill_between(
                bin_edges[:-1],
                ((mean_hist - std_hist) / (true_hist + 1e-20)),
                ((mean_hist + std_hist) / (true_hist + 1e-20)),
                step="post",
                alpha=0.3,
                color="tab:orange",
            )
            ax[1].step(
                bin_edges[:-1],
                (mean_hist / (true_hist + 1e-10)),
                color="tab:orange",
                where="post",
            )
            ax[1].axhline(y=1.0, color="black", linestyle="-")
            ax[1].axhline(y=1.2, color="tab:gray", linestyle="--")
            ax[1].axhline(y=0.8, color="tab:gray", linestyle="--")

            ax[2].set_ylabel(r"$\delta [\%]$")
            ax[2].set_xlabel("truth amplitudes")
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks(
                [
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                minor=True,
            )

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")

            ax[2].errorbar(
                bin_center, delta, color="k", linewidth=1.0, fmt=".", capsize=2
            )

            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.02, 0.02, 0.98, 0.98))
            pdf.savefig(fig)
            plt.close()

            # comparison delta (absolute value, full range)
            abs_delta_amp = np.fabs(delta_amp)
            d_lo = max(abs_delta_amp.min(), 1e-6)
            d_hi = abs_delta_amp.max()
            bins = np.logspace(np.log10(d_lo), np.log10(d_hi), 51)
            fig, axs = plt.subplots(1, 1, figsize=(7, 5))
            axs.hist(
                abs_delta_amp,
                histtype="step",
                bins=bins,
                density=True,
                label=self.label,
            )
            axs.set_yscale("log")
            axs.set_xscale("log")
            axs.set_xlabel(r"$|\Delta|$")
            axs.set_ylabel("normalized counts")
            axs.legend()
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.02, 0.02, 0.98, 0.98))
            pdf.savefig(fig)
            plt.close()

            # pull plot: delta / sigma
            pull_bins = np.linspace(
                np.quantile(pull, 0.005), np.quantile(pull, 0.995), 51
            )
            fig, axs = plt.subplots(1, 1, figsize=(7, 5))
            axs.hist(
                pull,
                histtype="step",
                bins=pull_bins,
                density=True,
                label=self.label,
                color="tab:blue",
            )
            x_range = np.linspace(pull_bins[0], pull_bins[-1], 200)
            axs.plot(
                x_range,
                stats.norm.pdf(x_range),
                color="tab:red",
                linestyle="--",
                label=r"$\mathcal{N}(0,1)$",
            )
            axs.set_xlabel(r"$\Delta / \sigma$")
            axs.set_ylabel("normalized counts")
            axs.legend()
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.02, 0.02, 0.98, 0.98))
            pdf.savefig(fig)
            plt.close()
