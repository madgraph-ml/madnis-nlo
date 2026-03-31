from __future__ import annotations
import warnings
import pickle
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker
from typing import Optional, Sequence, Any
from matplotlib.ticker import ScalarFormatter


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format her


@dataclass
class Line:
    y: np.ndarray
    y_err: np.ndarray | None = None
    line_ref: Optional[Line] = None
    label: str | None = None
    color: str | None = None
    linestyle: str | None = "solid"
    fill: bool = False
    vline: bool = False
    alpha: float = 1.0
    linewidth: float = 1.0


def force_logy_ticks(ax, *, max_major=3):
    """
    Robust log-y ticks:
      - minor ticks at 2..9 * 10^n within visible ylim
      - major ticks: <= max_major, always in-range; prefers decade ticks if possible
    Call AFTER all plotting and AFTER final ylim is known (autoscale or manual).
    """
    if ax.get_yscale() != "log":
        return

    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin <= 0 or ymax <= 0:
        return
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    logmin = np.log10(ymin)
    logmax = np.log10(ymax)

    e_lo = int(np.floor(logmin)) - 1
    e_hi = int(np.ceil(logmax)) + 1

    # -----------------
    # MINOR ticks: 2..9 * 10^n within [ymin, ymax]
    # -----------------
    minor = []
    for e in range(e_lo, e_hi + 1):
        decade = 10.0**e
        for s in range(2, 10):  # 2..9
            v = s * decade
            if ymin <= v <= ymax:
                minor.append(v)

    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # -----------------
    # MAJOR ticks: prefer decade ticks inside the view
    # -----------------
    decade_inside = []
    for e in range(e_lo, e_hi + 1):
        v = 10.0**e
        if ymin <= v <= ymax:
            decade_inside.append(v)

    decade_inside = np.array(decade_inside, dtype=float)

    if len(decade_inside) >= 2:
        # too many? subsample evenly down to max_major
        if len(decade_inside) > max_major:
            idx = np.linspace(0, len(decade_inside) - 1, max_major)
            major = decade_inside[np.round(idx).astype(int)]
        else:
            major = decade_inside

    elif len(decade_inside) == 1:
        # one decade tick in range: add endpoints (geometric) to avoid "single major tick" look
        candidates = np.unique(np.array([ymin, decade_inside[0], ymax], dtype=float))
        # if still too many, reduce to max_major by picking evenly in log space
        if len(candidates) > max_major:
            logc = np.log10(candidates)
            idx = np.linspace(0, len(candidates) - 1, max_major)
            major = candidates[np.round(idx).astype(int)]
        else:
            major = candidates

    else:
        # no decade tick inside: place majors geometrically within range
        n = min(max_major, 3)
        major = np.geomspace(ymin, ymax, n)

    ax.yaxis.set_major_locator(ticker.FixedLocator(major))
    # formatter that behaves nicely for both decades and non-decade majors
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))

    ax.minorticks_on()
    ax.tick_params(axis="y", which="minor", length=3)


def hist_plot(
    pdf: PdfPages,
    lines: list[Line],
    bins: np.ndarray,
    show_ratios: bool = True,
    absorb_reference_uncertainty_into_ratio: bool = False,
    show_sigma_deviations: bool = False,
    fks_hist: bool = False,  # whether it is the fks histogram, which has a different ylabel
    title: str | None = None,
    title_on_axs: bool = False,
    subtitle: str | None = None,
    no_scale: bool = False,
    xscale: str | None = None,
    yscale: str | None = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] | list[float, float] | None = None,
    model_name: str = "MadNIS",
    tex_label: str | None = None,
    unit: str | None = None,
    rect=(0.12, 0.12, 0.96, 0.96),
    size_multiplier: float = 1.0,
    legend_kwargs: dict | None = None,
    debug=False,
    ylabel: str | None = None,
):
    """
    Makes a single histogram plot, used for the observable histograms and clustering
    histograms.
    Args:
        pdf: Multipage PDF object
        lines: list of line objects describing the histograms
        bins: Numpy array with the bin boundaries
        show_ratios: If True, show a panel with ratios
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        n_panels = 1 + int(show_ratios) + int(show_sigma_deviations)
        height_ratios = [10]
        if show_ratios:
            height_ratios.append(4)
        if show_sigma_deviations:
            height_ratios.append(4)
        fig, axs = plt.subplots(
            n_panels,
            1,
            sharex=True,
            figsize=(6 * size_multiplier, 4.5 * size_multiplier),
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.00},
        )

        if n_panels == 1:
            axs = [axs]

        ratio_ax = None
        sigma_ax = None
        panel_idx = 1
        if show_ratios:
            ratio_ax = axs[panel_idx]
            panel_idx += 1
        if show_sigma_deviations:
            sigma_ax = axs[panel_idx]

        unc_band_shown = False
        one_sigma_band_shown = False
        for line in lines:
            if line.vline:
                axs[0].axvline(
                    line.y, label=line.label, color=line.color, linestyle=line.linestyle
                )
                continue
            integral = np.sum((bins[1:] - bins[:-1]) * line.y)
            scale = 1 / integral if integral != 0.0 else 1.0
            if line.line_ref is not None:
                ref_integral = np.sum((bins[1:] - bins[:-1]) * line.line_ref.y)
                ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
            if no_scale:
                scale = 1.0
                ref_scale = 1.0

            if debug:
                print("Actual values plotted:", line.y * scale)
            hist_line(
                axs[0],
                bins,
                line.y * scale,
                line.y_err * scale if line.y_err is not None else None,
                label=line.label,
                color=line.color,
                fill=line.fill,
                alpha=line.alpha,
                linestyle=line.linestyle,
                linewidth=line.linewidth,
            )

            if show_ratios and (line.line_ref is not None):
                ratio = (line.y * scale) / (line.line_ref.y * ref_scale)
                ratio_isnan = np.isnan(ratio)
                if line.y_err is not None:
                    if not absorb_reference_uncertainty_into_ratio:
                        if len(line.y_err.shape) == 2:
                            ratio_err = (line.y_err * scale) / (
                                line.line_ref.y * ref_scale
                            )
                            ratio_err[:, ratio_isnan] = 0.0
                        else:
                            ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                            ratio_err[ratio_isnan] = 0.0
                    else:
                        if line.line_ref.y_err is None:
                            raise ValueError(
                                "absorb_reference_uncertainty_into_ratio=True requires line.line_ref.y_err to be set"
                            )
                        if len(line.y_err.shape) == 2:
                            ratio_err = (
                                1
                                / (line.line_ref.y * ref_scale)
                                * np.sqrt(
                                    (line.y_err * scale) ** 2
                                    + (line.y * scale * line.line_ref.y_err * ref_scale)
                                    ** 2
                                    / (line.line_ref.y * ref_scale) ** 2
                                )
                            )
                            ratio_err[:, ratio_isnan] = 0.0
                        else:
                            ratio_err = (
                                1
                                / (line.line_ref.y * ref_scale)
                                * np.sqrt(
                                    (line.y_err * scale) ** 2
                                    + (line.y * scale * line.line_ref.y_err * ref_scale)
                                    ** 2
                                    / (line.line_ref.y * ref_scale) ** 2
                                )
                            )
                            ratio_err[ratio_isnan] = 0.0
                else:
                    ratio_err = None
                ratio[ratio_isnan] = 1.0
                hist_line(
                    ratio_ax,
                    bins,
                    ratio,
                    ratio_err,
                    label=None,
                    color=line.color,
                    alpha=line.alpha,
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                )
                if (
                    line.line_ref.y_err is not None
                    and not unc_band_shown
                    and not absorb_reference_uncertainty_into_ratio
                ):
                    unc_band = np.where(
                        np.isfinite(line.line_ref.y * ref_scale),
                        (line.line_ref.y_err * ref_scale) / (line.line_ref.y * ref_scale),
                        np.nan,
                    )
                    hist_line(
                        ratio_ax,
                        bins,
                        y=np.ones_like(unc_band),
                        y_err=unc_band,
                        label=None,
                        color=line.line_ref.color,
                        fill=False,
                        alpha=1,
                        linewidth=0.0,
                    )
                    unc_band_shown = True

            if show_sigma_deviations and (line.line_ref is not None):
                if line.line_ref.y_err is None:
                    raise ValueError(
                        "show_sigma_deviations=True requires line.line_ref.y_err to be set"
                    )
                y_scaled = line.y * scale
                yref_scaled = line.line_ref.y * ref_scale
                sigref_scaled = line.line_ref.y_err * ref_scale

                # N_sigma = (y - yref) / sigref
                nsig = np.zeros_like(y_scaled, dtype=float)
                bad = (
                    (~np.isfinite(y_scaled))
                    | (~np.isfinite(yref_scaled))
                    | (~np.isfinite(sigref_scaled))
                    | (sigref_scaled <= 0)
                )

                nsig[~bad] = (y_scaled[~bad] - yref_scaled[~bad]) / sigref_scaled[~bad]
                nsig[bad] = np.nan

                nsig_err = (
                    np.sqrt(line.y_err**2 + line.line_ref.y_err**2)
                    / line.line_ref.y_err
                )
                hist_line(
                    sigma_ax,
                    bins,
                    nsig,
                    y_err=nsig_err,
                    label=None,
                    color=line.color,
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    alpha=line.alpha,
                )
                one_sigma_band_shown = True
        if title is not None and not title_on_axs:
            corner_text(axs[0], title, "left", "top")
            if subtitle is not None:
                corner_text(axs[0], subtitle, "left", "top", is_subtitle=True)
        elif title is not None and title_on_axs:
            axs[0].set_title(title, loc="right")
        if legend_kwargs is None:
            axs[0].legend(loc="best", frameon=False, handlelength=1.0)
        else:
            lk = dict(legend_kwargs)  # copy, don't modify caller's dict
            hl = lk.pop("handlelength", 1.0)
            loc = lk.pop("loc", "best")
            axs[0].legend(loc=loc, frameon=False, handlelength=hl, **lk)

        deno = (
            r"p_{T}"
            if "p_{T," in tex_label
            else r"E"
            if "E" in tex_label
            else r"\Delta R"
            if r"\Delta R" in tex_label
            else r"\Delta \phi"
            if r"\Delta \phi" in tex_label
            else r"\Delta \eta"
            if r"\Delta \eta" in tex_label
            else r"m"
            if "m_" in tex_label
            else r"\eta"
            if r"\eta" in tex_label
            else r"\phi"
            if r"\phi" in tex_label
            else r"S_{T}"
            if "S_{T," in tex_label
            else "x"
        )
        unit = (
            unit
            if unit is not None
            else r"GeV"
            if "p_T" in tex_label
            or "E" in tex_label
            or "m_" in tex_label
            or "S_T" in tex_label
            else None
        )
        if unit is not None:
            ylabel = rf"$\mathrm{{d}}\sigma/\mathrm{{d}}{{{deno}}}\ [\mathrm{{pb}}/\mathrm{{{unit}}}]$"
        else:
            ylabel = rf"$\mathrm{{d}}\sigma/\mathrm{{d}}{{{deno}}}\ [\mathrm{{pb}}]$"
        if fks_hist:
            ylabel = r"$\mathrm{{d}}\sigma_{i,j}\ [\mathrm{{pb}}]$"
        axs[0].set_ylabel(ylabel)
        axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))  # 3 decimals
        if ylim:
            axs[0].set_ylim(ylim)
        else:
            axs[0].relim()
            axs[0].autoscale_view()
        if yscale == "log":
            axs[0].set_yscale(yscale)
            force_logy_ticks(axs[0], max_major=3)
        else:
            axs[0].set_yscale(yscale if yscale is not None else "linear")
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axs[0].yaxis.set_major_formatter(yfmt)
        axs[0].set_xscale(xscale if xscale is not None else "linear")

        if show_ratios:
            for l in lines:
                if l.line_ref is not None:
                    reference_label = l.line_ref.label
                    break

            # ratio_ax.set_ylabel(
            #     rf"$\frac{{\text{{{model_name}}}}}{{\text{{{reference_label}}}}}$"
            # )
            ratio_ax.set_ylabel(r"Ratio")
            ratio_ax.set_yticks([0.75, 1, 1.25])
            ratio_ax.set_ylim([0.5, 1.5])
            ratio_ax.axhline(y=1, c="black", ls="--", lw=0.7)
            ratio_ax.axhline(y=1.25, c="black", ls="dotted", lw=0.5)
            ratio_ax.axhline(y=0.75, c="black", ls="dotted", lw=0.5)
            ratio_ax.set_xscale(xscale if xscale is not None else "linear")

        if show_sigma_deviations:
            sigma_ax.set_ylabel(r"$\text{Dev. }[\sigma]$")
            sigma_ax.set_yticks([-2, -1, 0, 1, 2])
            sigma_ax.set_ylim([-3, 3])
            sigma_ax.axhline(y=0, c="black", ls="--", lw=0.7)
            sigma_ax.axhline(y=1, c="black", ls="dotted", lw=0.5)
            sigma_ax.axhline(y=-1, c="black", ls="dotted", lw=0.5)
            sigma_ax.axhline(y=2, c="black", ls="dotted", lw=0.3)
            sigma_ax.axhline(y=-2, c="black", ls="dotted", lw=0.3)
            sigma_ax.set_xscale(xscale if xscale is not None else "linear")

        unit = "" if unit is None else rf"$[\mathrm{{{unit}}}]$"
        if tex_label is not None:
            xlabel = rf"${tex_label}$ {unit}"
        else:
            xlabel = rf"Observable {unit}"
        axs[-1].set_xlabel(xlabel)
        if xlim is None:
            for ax in axs:
                ax.set_xlim(bins[0], bins[-1])
        else:
            for ax in axs:
                ax.set_xlim(xlim)

        if rect is not None:
            fig.subplots_adjust(
                left=rect[0],
                bottom=rect[1],
                right=rect[2],
                top=rect[3] - 0.02 * title_on_axs,
            )
        else:
            fig.subplots_adjust(
                left=0.12, bottom=0.12, right=0.96, top=0.96 - 0.02 * title_on_axs
            )
        fig.align_ylabels(axs)
        plt.savefig(pdf, format="pdf")
        plt.close()


def hist_line(
    ax: mpl.axes.Axes,
    bins: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "solid",
    fill: bool = False,
    alpha: float = 1.0,
    linewidth: float = 1.0,
):
    """
    Plot a stepped line for a histogram, optionally with error bars.
    Args:
        ax: Matplotlib Axes
        bins: Numpy array with bin boundaries
        y: Y values for the bins
        y_err: Y errors for the bins
        label: Label of the line
        color: Color of the line
        linestyle: line style
        fill: Filled histogram
    """

    dup_last = lambda a: np.append(a, a[-1])

    if fill:
        ax.fill_between(
            bins,
            dup_last(y),
            label=label,
            facecolor=color,
            step="post",
            alpha=0.2 * alpha,
        )
    else:
        ax.step(
            bins,
            dup_last(y),
            label=label,
            color=color,
            linewidth=linewidth,
            where="post",
            ls=linestyle,
            alpha=alpha,
        )
    if y_err is not None:
        if len(y_err.shape) == 2:
            y_low = y_err[0]
            y_high = y_err[1]
        else:
            y_low = y - y_err
            y_high = y + y_err

        ax.step(
            bins,
            dup_last(y_high),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5 * linewidth,
            where="post",
        )
        ax.step(
            bins,
            dup_last(y_low),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5 * linewidth,
            where="post",
        )
        ax.fill_between(
            bins,
            dup_last(y_low),
            dup_last(y_high),
            facecolor=color,
            alpha=0.3 * alpha,
            step="post",
        )


def compute_hist_data_simple_histogram(
    bins: np.ndarray,
    data: np.ndarray,
    bayesian=False,
    weights=None,
):
    if bayesian:
        hists = np.stack(
            [np.histogram(d, bins=bins, density=False, weights=weights)[0] for d in data],
            axis=0,
        )
        y = hists[0]
        y_err = np.std(hists, axis=0)
    else:
        y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
        n = len(weights)
        y = y / n
        y_err_sq, _ = np.histogram(data, bins=bins, density=False, weights=weights**2)
        y_err = np.sqrt(y_err_sq) / n
    return y, y_err


def compute_hist_data(
    bins: np.ndarray,
    data_n: np.ndarray,
    data_np1: np.ndarray,
    weights_n=None,
    weights_np1=None,
    valid_n=None,
    valid_np1=None,
    bayesian=False,
    debug=False,
):
    if bayesian:
        raise NotImplementedError("Bayesian mode not supported here")
    bin_idx_n = np.digitize(data_n, bins) - 1
    bin_idx_np1 = np.digitize(data_np1, bins) - 1
    nbins = len(bins) - 1

    valid_n = (bin_idx_n >= 0) & (bin_idx_n < nbins) & valid_n
    valid_np1 = (bin_idx_np1 >= 0) & (bin_idx_np1 < nbins) & valid_np1

    # weight contributions
    sum_w = np.zeros(nbins, dtype=float)
    sum_w2 = np.zeros(nbins, dtype=float)

    same_bin = valid_n & valid_np1 & (bin_idx_n == bin_idx_np1)

    # Both in same bin: combined weight
    w_same = weights_n[same_bin] + weights_np1[same_bin]
    idx_same = bin_idx_n[same_bin]
    np.add.at(sum_w, idx_same, w_same)
    np.add.at(sum_w2, idx_same, w_same**2)

    # Different bins: fill once
    n_only = valid_n & ~same_bin
    np1_only = valid_np1 & ~same_bin

    np.add.at(sum_w, bin_idx_n[n_only], weights_n[n_only])
    np.add.at(sum_w2, bin_idx_n[n_only], weights_n[n_only] ** 2)

    np.add.at(sum_w, bin_idx_np1[np1_only], weights_np1[np1_only])
    np.add.at(sum_w2, bin_idx_np1[np1_only], weights_np1[np1_only] ** 2)

    bin_widths = bins[1:] - bins[:-1]
    y = sum_w / bin_widths

    # y_err = np.sqrt(sum_w2) / bin_widths      # RMS error
    var = sum_w2 - (sum_w**2) / len(data_n)
    y_err = np.sqrt(var) / bin_widths  # Standard deviation error

    if debug:
        print(
            f"DEBUG: Integral of the histogram is {np.sum((bins[1:] - bins[:-1]) * y):.5f} +/- {np.sqrt(np.sum(( (bins[1:] - bins[:-1]) * y_err )**2)):.5f}"
        )
    return y, y_err


def corner_text(
    ax: mpl.axes.Axes,
    text: str,
    horizontal_pos: str,
    vertical_pos: str,
    is_subtitle: bool = False,
):
    ax.text(
        x=0.95 if horizontal_pos == "right" else 0.05,
        y=(
            0.95 - 0.1 * (is_subtitle)
            if vertical_pos == "top"
            else 0.05 + 0.1 * (is_subtitle)
        ),
        s=text,
        horizontalalignment=horizontal_pos,
        verticalalignment=vertical_pos,
        transform=ax.transAxes,
    )
    # Dummy line for automatic legend placement
    plt.plot(
        0.8 if horizontal_pos == "right" else 0.2,
        0.8 - 0.1 * (is_subtitle) if vertical_pos == "top" else 0.2 + 0.1 * (is_subtitle),
        transform=ax.transAxes,
        color="none",
    )


def append_to_pickle(new_entry, pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        pickle_data = []

    pickle_data.append(new_entry)
    with open(pickle_file, "wb") as f:
        pickle.dump(pickle_data, f)
