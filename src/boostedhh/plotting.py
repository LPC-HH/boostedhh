"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from hist.intervals import poisson_interval
from numpy.typing import ArrayLike
from pandas import DataFrame

from boostedhh import utils
from boostedhh.hh_vars import LUMI, data_key, hbb_bg_keys
from boostedhh.utils import CUT_MAX_VAL

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

# Data point styling parameters
DATA_STYLE = {
    "histtype": "errorbar",
    "color": "black",
    "markersize": 15,
    "elinewidth": 2,
    "capsize": 0,
}

# this is needed for some reason to update the font size for the first plot
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# plt.rcParams.update({"font.size": 24})
# plt.close()

BG_UNC_LABEL = "Total Bkg. Uncertainty"

sample_label_map = {
    "ST": r"Single-$t$",
    "TT": r"$t\bar{t}$",
    "Hbb": r"H$\rightarrow b\overline{b}$",
}

COLOURS = {
    # CMS 10-colour-scheme from
    # https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/#categorical-data-eg-1d-stackplots
    "darkblue": "#3f90da",
    "lightblue": "#92dadd",
    "orange": "#e76300",
    "red": "#bd1f01",
    "darkpurple": "#832db6",
    "brown": "#a96b59",
    "gray": "#717581",
    "beige": "#b9ac70",
    "yellow": "#ffa90e",
    "lightgray": "#94a4a2",
    # extra colours
    "darkred": "#A21315",
    "green": "#7CB518",
    "mantis": "#81C14B",
    "forestgreen": "#2E933C",
    "darkgreen": "#064635",
    "purple": "#9381FF",
    "deeppurple": "#36213E",
    "ashgrey": "#ACBFA4",
    "canary": "#FFE51F",
    "arylideyellow": "#E3C567",
    "earthyellow": "#D9AE61",
    "satinsheengold": "#C8963E",
    "flax": "#EDD382",
    "vanilla": "#F2F3AE",
    "dutchwhite": "#F5E5B8",
}

MARKERS = [
    "o",
    "^",
    "v",
    "<",
    ">",
    "s",
    "+",
    "x",
    "d",
    "1",
    "2",
    "3",
    "4",
    "h",
    "p",
    "|",
    "_",
    "D",
    "H",
]

# for more than 5, probably better to use different MARKERS
LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 5, 1, 5, 1, 5)),
]


BG_COLOURS = {
    "QCD": "darkblue",
    "TT": "brown",
    "W+Jets": "orange",
    "Z+Jets": "yellow",
    "ST": "lightblue",
    "Diboson": "lightgray",
    "Hbb": "beige",
    "HWW": "gray",
}

sig_colour = "red"

SIG_COLOURS = [
    "#bd1f01",
    "#ff5252",
    "#7F2CCB",
    "#ffbaba",
    # "#ff7b7b",
    "#885053",
    "#a70000",
    "#5A1807",
    "#3C0919",
    "#353535",
]

ROC_COLOURS = [
    "darkblue",
    "lightblue",
    "orange",
    "brown",
    "darkpurple",
    "red",
    "gray",
    "beige",
    "yellow",
]


def _combine_hbb_bgs(hists, bg_keys):
    """combine all hbb backgrounds into a single "Hbb" background for plotting"""

    # skip this if no hbb bg keys specified
    if len(set(bg_keys) & set(hbb_bg_keys)) == 0:
        return hists, bg_keys

    h = utils.combine_hbb_bgs(hists)

    bg_keys = [key for key in bg_keys if key not in hbb_bg_keys]

    if "Hbb" not in bg_keys:
        bg_keys.append("Hbb")

    return h, bg_keys


def _process_samples(
    sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation, sample_label_map
):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [COLOURS[bg_colours[sample]] for sample in bg_keys]
    bg_labels = [sample_label_map.get(bg_key, bg_key) for bg_key in bg_keys]

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = {key: val for key, val in sig_scale_dict.items() if key in sig_keys}

    sig_labels = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = sample_label_map.get(sig_key, sig_key)

        if sig_scale != 1:
            if sig_scale <= 10000:
                label = f"{label} $\\times$ {sig_scale:.0f}"
            else:
                label = f"{label} $\\times$ {sig_scale:.1e}"

        sig_labels[sig_key] = label

    # set up systematic variations if needed
    if syst is not None and variation is not None:
        wshift, wsamples = syst
        shift = variation
        skey = {"up": " Up", "down": " Down"}[shift]

        for i, key in enumerate(bg_keys):
            if key in wsamples:
                bg_keys[i] += f"_{wshift}_{shift}"
                bg_labels[i] += skey

        for sig_key in list(sig_scale_dict.keys()):
            if sig_key in wsamples:
                new_key = f"{sig_key}_{wshift}_{shift}"
                sig_scale_dict[new_key] = sig_scale_dict[sig_key]
                sig_labels[new_key] = sig_labels[sig_key] + skey
                del sig_scale_dict[sig_key], sig_labels[sig_key]

    return bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels


def _divide_bin_widths(hists, data_err, bg_tot, bg_err):
    """Divide histograms by bin widths"""
    edges = hists.axes[1].edges
    bin_widths = edges[1:] - edges[:-1]

    if data_err is None:
        data_err = (
            np.abs(poisson_interval(hists[data_key, ...].values()) - hists[data_key, ...].values())
            / bin_widths
        )

    if bg_err is not None:
        bg_err = bg_err / bin_widths

    bg_tot = bg_tot / bin_widths
    hists = hists / bin_widths[np.newaxis, :]
    return hists, data_err, bg_tot, bg_err


def _fill_error(ax, edges, down, up, scale=1):
    ax.fill_between(
        np.repeat(edges, 2)[1:-1],
        np.repeat(down, 2) * scale,
        np.repeat(up, 2) * scale,
        color="black",
        alpha=0.2,
        hatch="//",
        linewidth=0,
    )


def _asimov_significance(s, b):
    """Asimov estimate of discovery significance (with no systematic uncertainties).
    See e.g. https://www.pp.rhul.ac.uk/~cowan/atlas/cowan_atlas_15feb11.pdf.
    Or for more explanation: https://www.pp.rhul.ac.uk/~cowan/stat/cowan_munich16.pdf
    """
    return np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))


def add_cms_label(ax, year, data=True, label="Preliminary", loc=2, lumi=True):
    if year == "all":
        hep.cms.label(
            label,
            data=data,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}" if lumi else None,
            year=None,
            ax=ax,
            loc=loc,
        )
    else:
        hep.cms.label(
            label,
            data=data,
            lumi=f"{LUMI[year] / 1e3:.0f}" if lumi else None,
            year=year,
            ax=ax,
            loc=loc,
        )


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: list[str],
    bg_keys: list[str],
    sig_colours: list[str] = None,
    bg_colours: dict[str, str] = None,
    sig_err: ArrayLike | str = None,
    bg_err: ArrayLike = None,
    data_err: ArrayLike | bool | None = None,
    title: str = None,
    name: str = "",
    sig_scale_dict: OrderedDict[str, float] = None,
    ylim: int = None,
    show: bool = True,
    syst: tuple = None,
    variation: str = None,
    region_label: str = None,
    sample_label_map: dict[str, str] = sample_label_map,
    bg_err_type: str = "shaded",
    plot_data: bool = True,
    bg_order: list[str] = None,
    log: bool = False,
    ratio_ylims: list[float] = None,
    divide_bin_width: bool = False,
    plot_significance: bool = False,
    significance_dir: str = "right",
    plot_ratio: bool = True,
    axraxsax: tuple = None,
    leg_args: dict = None,
    cmslabel: str = None,
    cmsloc: int = 0,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below

    Args:
        hists (Hist): input histograms per sample to plot
        year (str): datataking year
        sig_keys (List[str]): signal keys
        bg_keys (List[str]): background keys
        sig_colours (Dict[str, str], optional): dictionary of colours per signal. Defaults to sig_colours.
        bg_colours (Dict[str, str], optional): dictionary of colours per background. Defaults to bg_colours.
        sig_err (Union[ArrayLike, str], optional): plot error on signal.
          if string, will take up down shapes from the histograms (assuming they're saved as "{sig_key}_{sig_err}_{up/down}")
          if 1D Array, will take as error per bin
        bg_err (ArrayLike, optional): [bg_tot_down, bg_tot_up] to plot bg variations. Defaults to None.
        data_err (Union[ArrayLike, bool, None], optional): plot error on data.
          if True, will plot poisson error per bin
          if array, will plot given errors per bins
        title (str, optional): plot title. Defaults to None.
        name (str): name of file to save plot
        sig_scale_dict (Dict[str, float]): if scaling signals in the plot, dictionary of factors
          by which to scale each signal
        ylim (optional): y-limit on plot
        show (bool): show plots or not
        syst (Tuple): Tuple of (wshift: name of systematic e.g. pileup,  wsamples: list of samples which are affected by this),
          to plot variations of this systematic.
        variation (str): options:
          "up" or "down", to plot only one wshift variation (if syst is not None).
          Defaults to None i.e. plotting both variations.
        bg_err_type (str): "shaded" or "line".
        plot_data (bool): plot data
        bg_order (List[str]): order in which to plot backgrounds
        ratio_ylims (List[float]): y limits on the ratio plots
        divide_bin_width (bool): divide yields by the bin width (for resonant fit regions)
        plot_significance (bool): plot Asimov significance below ratio plot
        significance_dir (str): "Direction" for significance. i.e. a > cut ("right"), a < cut ("left"), or per-bin ("bin").
        axrax (Tuple): optionally input ax and rax instead of creating new ones
        ncol (int): # of legend columns. By default, it is 2 for log-plots and 1 for non-log-plots.
    """

    if ratio_ylims is None:
        ratio_ylims = [0, 2]
    if bg_colours is None:
        bg_colours = BG_COLOURS
    if sig_colours is None:
        sig_colours = SIG_COLOURS
    if leg_args is None:
        leg_args = {"fontsize": 24}
    leg_args["ncol"] = leg_args.get("ncol", (2 if log else 1))

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    hists, bg_keys = _combine_hbb_bgs(hists, bg_keys)
    data_label = sample_label_map.get(data_key, data_key)

    bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels = _process_samples(
        sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation, sample_label_map
    )

    bg_tot = np.maximum(sum([hists[sample, :] for sample in bg_keys]).values(), 0.0)

    if syst is not None and variation is None:
        # plot up/down variations
        wshift, wsamples = syst
        if sig_keys[0] in wsamples:
            sig_err = wshift  # will plot sig variations below
        bg_err = []
        for shift in ["down", "up"]:
            bg_sums = []
            for sample in bg_keys:
                if sample in wsamples and f"{sample}_{wshift}_{shift}" in hists.axes[0]:
                    bg_sums.append(hists[f"{sample}_{wshift}_{shift}", :].values())
                # elif sample != "Hbb":
                else:
                    bg_sums.append(hists[sample, :].values())
            bg_err.append(np.maximum(np.sum(bg_sums, axis=0), 0.0))
        bg_err = np.array(bg_err)

    pre_divide_hists = hists
    pre_divide_bg_tot = bg_tot

    if divide_bin_width:
        hists, data_err, bg_tot, bg_err = _divide_bin_widths(hists, data_err, bg_tot, bg_err)

    # set up plots
    if axraxsax is not None:
        if plot_significance:
            ax, rax, sax = axraxsax
        elif plot_ratio:
            ax, rax = axraxsax
        else:
            ax = axraxsax

    elif plot_significance:
        fig, (ax, rax, sax) = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.1},
            sharex=True,
        )
    elif plot_ratio:
        fig, (ax, rax) = plt.subplots(
            2,
            1,
            figsize=(12, 14),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    plt.rcParams.update({"font.size": 24})

    # plot histograms
    y_label = r"Events / GeV" if divide_bin_width else "Events"
    ax.set_ylabel(y_label)

    # background samples
    if len(bg_keys):
        hep.histplot(
            [hists[sample, :] for sample in bg_keys],
            ax=ax,
            histtype="fill",
            stack=True,
            label=bg_labels,
            color=bg_colours,
            flow="none",
        )

    # signal samples
    if len(sig_scale_dict):
        hep.histplot(
            [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            label=list(sig_labels.values()),
            color=sig_colours[: len(sig_keys)],
            linewidth=3,
            flow="none",
        )

        # plot signal errors
        if isinstance(sig_err, str):
            for skey, shift in [("Up", "up"), ("Down", "down")]:
                hep.histplot(
                    [
                        hists[f"{sig_key}_{sig_err}_{shift}", :] * sig_scale
                        for sig_key, sig_scale in sig_scale_dict.items()
                    ],
                    yerr=0,
                    ax=ax,
                    histtype="step",
                    label=[f"{sig_label} {skey}" for sig_label in sig_labels.values()],
                    alpha=0.6,
                    color=sig_colours[: len(sig_keys)],
                    flow="none",
                )
        elif sig_err is not None:
            for sig_key, sig_scale in sig_scale_dict.items():
                _fill_error(
                    ax,
                    hists.axes[1].edges,
                    hists[sig_key, :].values() * (1 - sig_err),
                    hists[sig_key, :].values() * (1 + sig_err),
                    sig_scale,
                )

    if bg_err is not None:
        # if divide_bin_width:
        #     raise NotImplementedError("Background error for divide bin width not checked yet")

        if len(np.array(bg_err).shape) == 1:
            bg_err = [bg_tot - bg_err, bg_tot + bg_err]

        if bg_err_type == "shaded":
            ax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat(bg_err[0], 2),
                np.repeat(bg_err[1], 2),
                color="black",
                alpha=0.2,
                hatch="//",
                linewidth=0,
                label=BG_UNC_LABEL,
            )
        else:
            ax.stairs(
                bg_tot,
                hists.axes[1].edges,
                color="black",
                linewidth=3,
                label="BG Total",
                baseline=bg_tot,
            )

            ax.stairs(
                bg_err[0],
                hists.axes[1].edges,
                color="red",
                linewidth=3,
                label="BG Down",
                baseline=bg_err[0],
            )

            ax.stairs(
                bg_err[1],
                hists.axes[1].edges,
                color="#7F2CCB",
                linewidth=3,
                label="BG Up",
                baseline=bg_err[1],
            )

    # plot data
    if plot_data:
        hep.histplot(
            hists[data_key, :],
            ax=ax,
            yerr=data_err,
            xerr=divide_bin_width,
            label=data_label,
            **DATA_STYLE,
            flow="none",
        )

    # legend ordering
    legend_order = [data_label] + bg_order[::-1] + list(sig_labels.values()) + [BG_UNC_LABEL]
    legend_order = [sample_label_map.get(k, k) for k in legend_order]

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[labels.index(label)] for label in legend_order if label in labels]
    ordered_labels = [label for label in legend_order if label in labels]
    ax.legend(ordered_handles, ordered_labels, **leg_args)

    if log:
        ax.set_yscale("log")

    y_lowlim = 0 if not log else 1e-5
    if ylim is not None:
        ax.set_ylim([y_lowlim, ylim])
    else:
        ax.set_ylim(y_lowlim, ax.get_ylim()[1] * 2)

    ax.margins(x=0)

    # plot ratio below
    if plot_ratio:
        if plot_data:
            # new: plotting data errors (black lines) and background errors (shaded) separately
            yerr = np.nan_to_num(
                np.abs(
                    poisson_interval(pre_divide_hists[data_key, ...].values())
                    - pre_divide_hists[data_key, ...].values()
                )
                / (pre_divide_bg_tot + 1e-5)
            )

            hep.histplot(
                pre_divide_hists[data_key, :] / (pre_divide_bg_tot + 1e-5),
                yerr=yerr,
                xerr=divide_bin_width,
                ax=rax,
                **DATA_STYLE,
                flow="none",
            )

            if bg_err is not None and bg_err_type == "shaded":
                # (bkg + err) / bkg
                rax.fill_between(
                    np.repeat(hists.axes[1].edges, 2)[1:-1],
                    np.repeat((bg_err[0]) / bg_tot, 2),
                    np.repeat((bg_err[1]) / bg_tot, 2),
                    color="black",
                    alpha=0.1,
                    hatch="//",
                    linewidth=0,
                )
        else:
            rax.set_xlabel(hists.axes[1].label)

        rax.set_ylabel("Data / Bkg.")
        rax.set_ylim(ratio_ylims)
        rax.grid()
        rax.margins(x=0)

        ax.set_xlabel(None)

    if plot_significance:
        sigs = [pre_divide_hists[sig_key, :].values() for sig_key in sig_scale_dict]

        if significance_dir == "left":
            bg_tot = np.cumsum(bg_tot[::-1])[::-1]
            sigs = [np.cumsum(sig[::-1])[::-1] for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\leq$ Cuts")
        elif significance_dir == "right":
            bg_tot = np.cumsum(bg_tot)
            sigs = [np.cumsum(sig) for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\geq$ Cuts")
        elif significance_dir == "bin":
            sax.set_ylabel("Asimov Sign. per Bin")
        else:
            raise RuntimeError(
                'Invalid value for ``significance_dir``. Options are ["left", "right", "bin"].'
            )

        edges = pre_divide_hists.axes[1].edges
        hep.histplot(
            [(_asimov_significance(sig, bg_tot), edges) for sig in sigs],
            ax=sax,
            histtype="step",
            label=[sample_label_map.get(sig_key, sig_key) for sig_key in sig_scale_dict],
            color=sig_colours[: len(sig_keys)],
            flow="none",
        )

        sax.legend(fontsize=15)
        sax.set_yscale("log")
        sax.set_ylim([1e-7, 10])
        sax.set_xlabel(hists.axes[1].label)
        sax.set_ylabel(sax.get_ylabel(), fontsize=22)
        rax.set_xlabel(None)

    if title is not None:
        ax.set_title(title, y=1.08)

    if region_label is not None:
        mline = "\n" in region_label
        xpos = 0.29 if not mline else 0.24
        ypos = 0.915 if not mline else 0.87
        ax.text(
            xpos,
            ypos,
            region_label,
            transform=ax.transAxes,
            fontsize=24,
            fontproperties="Tex Gyre Heros:bold",
        )

    add_cms_label(ax, year, label=cmslabel, loc=cmsloc)

    if axraxsax is None:
        if len(name):
            plt.savefig(name, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def sigErrRatioPlot(
    h: Hist,
    year: str,
    sig_key: str,
    wshift: str,
    title: str = None,
    plot_dir: str = None,
    name: str = None,
    show: bool = False,
):
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    nom = h[sig_key, :].values()
    hep.histplot(
        h[sig_key, :],
        histtype="step",
        label=sig_key,
        yerr=False,
        color=SIG_COLOURS[0],
        ax=ax,
        linewidth=2,
    )

    for skey, shift in [("Up", "up"), ("Down", "down")]:
        colour = {"up": "#81C14B", "down": "#1f78b4"}[shift]
        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :],
            histtype="step",
            yerr=False,
            label=f"{sig_key} {skey}",
            color=colour,
            ax=ax,
            linewidth=2,
        )

        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :] / nom,
            histtype="errorbar",
            # yerr=False,
            label=f"{sig_key} {skey}",
            color=colour,
            ax=rax,
        )

    ax.legend()
    ax.set_ylim(0)
    ax.set_ylabel("Events")
    add_cms_label(ax, year)
    ax.set_title(title, y=1.08)

    rax.set_ylim([0, 2])
    rax.set_xlabel(r"$m^{bb}_{reg}$ (GeV)")
    rax.legend()
    rax.set_ylabel("Variation / Nominal")
    rax.grid(axis="y")

    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def rocCurve(
    fpr,
    tpr,
    auc=None,
    sig_eff_lines=None,
    # bg_eff_lines=[],
    title=None,
    xlim=None,
    ylim=None,
    plot_dir="",
    name="",
    log: bool = True,
    show: bool = False,
):
    """Plots a ROC curve"""
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 0.8]
    if sig_eff_lines is None:
        sig_eff_lines = []

    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))

    plt.plot(tpr, fpr, label=f"AUC: {auc:.2f}" if auc is not None else None)

    for sig_eff in sig_eff_lines:
        y = fpr[np.searchsorted(tpr, sig_eff)]
        plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
        plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    if log:
        plt.yscale("log")

    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.title(title)
    plt.grid(which="major")

    if auc is not None:
        plt.legend()

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    hep.cms.label(data=False, label="Preliminary", rlabel="(13 TeV)")

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def multiROCCurveGreyOld(
    rocs: dict,
    sig_effs: list[float],
    plot_dir: Path,
    xlim=None,
    ylim=None,
    name: str = "",
    log: bool = True,
    show: bool = False,
):
    """_summary_

    Args:
        rocs (dict): {label: {sig_key1: roc, sig_key2: roc, ...}, ...} where label is e.g Test or Train
        sig_effs (list[float]): plot signal efficiency lines
    """
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))
    for roc_sigs in rocs.values():
        for roc in roc_sigs.values():
            auc_label = f" (AUC: {roc['auc']:.2f})" if "auc" in roc else ""

            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"] + auc_label,
                linewidth=2,
            )

            for sig_eff in sig_effs:
                y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
                plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
                plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    hep.cms.label(data=False, label="Preliminary", rlabel="(13 TeV)")
    if log:
        plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="upper left")
    plt.grid(which="major")

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


th_colours = [
    "#36213E",
    # "#9381FF",
    "#1f78b4",
    "#a6cee3",
    # "#32965D",
    "#7CB518",
    "#EDB458",
    "#ff7f00",
    "#a70000",
]


def multiROCCurveGrey(
    rocs: dict,
    sig_effs: list[float] = None,
    bkg_effs: list[float] = None,
    xlim=None,
    ylim=None,
    plot_dir: Path = None,
    name: str = "",
    show: bool = False,
    add_cms_label=False,
    legtitle: str = None,
    title: str = None,
    plot_thresholds: dict = None,  # plot signal and bkg efficiency for a given discriminator threshold
    find_from_sigeff: dict = None,  # find discriminator threshold that matches signal efficiency
):
    """Plot multiple ROC curves (e.g. train and test) + multiple signals"""
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}
    th_colours = ["cornflowerblue", "deepskyblue", "mediumblue", "cyan", "cadetblue"]
    eff_colours = ["lime", "aquamarine", "greenyellow"]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    for roc_sigs in rocs.values():

        # plots roc curves for each type of signal
        for roc in roc_sigs.values():

            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                color=roc["color"],
                linewidth=2,
            )

            # determines the point on the ROC curve that corresponds to the signal efficiency
            # plots a vertical and horizontal line to the point
            if sig_effs is not None:
                for sig_eff in sig_effs:
                    y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
                    plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
                    plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

            # determines the point on the ROC curve that corresponds to the background efficiency
            # plots a vertical and horizontal line to the point
            if bkg_effs is not None:
                for bkg_eff in bkg_effs:
                    x = roc["tpr"][np.searchsorted(roc["fpr"], bkg_eff)]
                    plt.vlines(x=x, ymin=0, ymax=bkg_eff, **line_style)
                    plt.hlines(y=bkg_eff, xmin=0, xmax=x, **line_style)

    # plots points and lines on plot corresponding to classifier thresholds
    for roc_sigs in rocs.values():
        if plot_thresholds is None:
            break
        i_sigeff = 0
        i_th = 0
        for rockey, roc in roc_sigs.items():
            if rockey in plot_thresholds:
                pths = {th: [[], []] for th in plot_thresholds[rockey]}
                for th in plot_thresholds[rockey]:
                    idx = _find_nearest(roc["thresholds"], th)
                    pths[th][0].append(roc["tpr"][idx])
                    pths[th][1].append(roc["fpr"][idx])
                for th in plot_thresholds[rockey]:
                    plt.scatter(
                        *pths[th],
                        marker="o",
                        s=40,
                        label=rf"{rockey} > {th:.2f}",
                        zorder=100,
                        color=th_colours[i_th],
                    )
                    plt.vlines(
                        x=pths[th][0],
                        ymin=0,
                        ymax=pths[th][1],
                        color=th_colours[i_th],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    plt.hlines(
                        y=pths[th][1],
                        xmin=0,
                        xmax=pths[th][0],
                        color=th_colours[i_th],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    i_th += 1

            if find_from_sigeff is not None and rockey in find_from_sigeff:
                pths = {sig_eff: [[], []] for sig_eff in find_from_sigeff[rockey]}
                thrs = {}
                for sig_eff in find_from_sigeff[rockey]:
                    idx = _find_nearest(roc["tpr"], sig_eff)
                    thrs[sig_eff] = roc["thresholds"][idx]
                    pths[sig_eff][0].append(roc["tpr"][idx])
                    pths[sig_eff][1].append(roc["fpr"][idx])
                for sig_eff in find_from_sigeff[rockey]:
                    plt.scatter(
                        *pths[sig_eff],
                        marker="o",
                        s=40,
                        label=rf"{rockey} > {thrs[sig_eff]:.2f}",
                        zorder=100,
                        color=eff_colours[i_sigeff],
                    )
                    plt.vlines(
                        x=pths[sig_eff][0],
                        ymin=0,
                        ymax=pths[sig_eff][1],
                        color=eff_colours[i_sigeff],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    plt.hlines(
                        y=pths[sig_eff][1],
                        xmin=0,
                        xmax=pths[sig_eff][0],
                        color=eff_colours[i_sigeff],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    i_sigeff += 1

    if add_cms_label:
        hep.cms.label(data=False, rlabel="")
    if title:
        plt.title(title)
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    if legtitle:
        plt.legend(title=legtitle, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if len(name):
        plt.savefig(plot_dir / f"{name}.png", bbox_inches="tight")
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def multiROCCurve(
    rocs: dict,
    thresholds=None,
    title=None,
    xlim=None,
    ylim=None,
    log=True,
    year="all",
    kin_label=None,
    plot_dir="",
    name="",
    prelim=True,
    lumi=None,
    show=False,
):
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    if thresholds is None:
        thresholds = [
            0.9,
            0.99,
            0.998,
        ]  # [0.9, 0.98, 0.995, 0.9965, 0.998] # [[0.99, 0.997, 0.998, 0.999, 0.9997]]

    plt.rcParams.update({"font.size": 32})

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, roc_sigs in enumerate(rocs.values()):
        for j, roc in enumerate(roc_sigs.values()):
            if len(np.array(thresholds).shape) > 1:
                pthresholds = thresholds[j]
            else:
                pthresholds = thresholds

            ax.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                linewidth=3,
                color=COLOURS[ROC_COLOURS[i * len(roc_sigs) + j]],
                linestyle=LINESTYLES[i * len(roc_sigs) + j],
            )
            pths = {th: [[], []] for th in pthresholds}
            for th in pthresholds:
                idx = _find_nearest(roc["thresholds"], th)
                pths[th][0].append(roc["tpr"][idx])
                pths[th][1].append(roc["fpr"][idx])
                # print(roc["tpr"][idx])

            for k, th in enumerate(pthresholds):
                ax.scatter(
                    *pths[th],
                    marker="o",
                    s=80,
                    label=(
                        f"Score > {th}" if i == len(rocs) - 1 and j == len(roc_sigs) - 1 else None
                    ),
                    color=th_colours[k],
                    zorder=100,
                )

                ax.vlines(
                    x=pths[th][0],
                    ymin=0,
                    ymax=pths[th][1],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

                ax.hlines(
                    y=pths[th][1],
                    xmin=0,
                    xmax=pths[th][0],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

                hep.cms.label(
                    ax=ax,
                    label="Preliminary" if prelim else "",
                    data=True,
                    year=year,
                    com="13.6",
                    fontsize=20,
                    lumi=lumi,
                )

    if log:
        plt.yscale("log")

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=21)

    if title:
        ax.text(
            0.02,
            0.93,
            title,
            transform=ax.transAxes,
            fontsize=20,
            # fontproperties="Tex Gyre Heros:bold",
        )

    if kin_label:
        ax.text(
            0.05,
            0.72,
            kin_label,
            transform=ax.transAxes,
            fontsize=20,
            fontproperties="Tex Gyre Heros",
        )

    if len(name):
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")
        plt.savefig(f"{plot_dir}/{name}.png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def ratioTestTrain(
    h: Hist,
    training_keys: list[str],
    shape_var: utils.ShapeVar,
    year: str,
    plot_dir="",
    name="",
    show=False,
):
    """Line and ratio plots comparing training and testing distributions

    Args:
        h (Hist): Histogram with ["Train", "Test"] x [sample] x [shape_var] axes
        training_keys (List[str]): List of training samples.
        shape_var (utils.ShapeVar): Variable being plotted.
        year (str): year.
    """
    plt.rcParams.update({"font.size": 24})

    style = {
        "Train": {"linestyle": "--"},
        "Test": {"alpha": 0.5},
    }

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    labels = [sample_label_map.get(key, key) for key in training_keys]
    for data in ["Train", "Test"]:
        plot_hists = [h[data, sample, :] for sample in training_keys]

        ax.set_ylabel("Events")
        hep.histplot(
            plot_hists,
            ax=ax,
            histtype="step",
            label=[data + " " + label for label in labels],
            color=[COLOURS[BG_COLOURS[sample]] for sample in training_keys],
            yerr=True,
            **style[data],
        )

    ax.set_xlim([shape_var.axis.edges[0], shape_var.axis.edges[-1]])
    ax.set_yscale("log")
    ax.legend(fontsize=20, ncol=2, loc="center left")

    plot_hists = [h["Train", sample, :] / h["Test", sample, :].values() for sample in training_keys]
    err = [
        np.sqrt(
            np.sum(
                [
                    h[data, sample, :].variances() / (h[data, sample, :].values() ** 2)
                    for data in ["Train", "Test"]
                ],
                axis=0,
            )
        )
        for sample in training_keys
    ]

    hep.histplot(
        plot_hists,
        ax=rax,
        histtype="errorbar",
        label=labels,
        color=[COLOURS[BG_COLOURS[sample]] for sample in training_keys],
        yerr=np.abs([err[i] * plot_hists[i].values() for i in range(len(plot_hists))]),
    )

    rax.set_ylim([0, 2])
    rax.set_xlabel(shape_var.label)
    rax.set_ylabel("Train / Test")
    rax.legend(fontsize=20, loc="upper left", ncol=3)
    rax.grid()

    hep.cms.label(data=False, year=year, ax=ax, lumi=f"{LUMI[year] / 1e3:.0f}")

    if len(name):
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def cutsLinePlot(
    events_dict: dict[str, DataFrame],
    shape_var: utils.ShapeVar,
    plot_key: str,
    cut_var: str,
    cut_var_label: str,
    cuts: list[float],
    year: str,
    weight_key: str,
    bb_masks: dict[str, DataFrame] = None,
    plot_dir: str = "",
    name: str = "",
    ratio: bool = False,
    ax: plt.Axes = None,
    show: bool = False,
):
    """Plot line plots of ``shape_var`` for different cuts on ``cut_var``."""
    if ax is None:
        if ratio:
            assert cuts[0] == 0, "First cut must be 0 for ratio plots."
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(12, 14),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        in_ax = False
    else:
        if ratio:
            raise NotImplementedError("Ratio plots not implemented with input axes.")
        in_ax = True

    plt.rcParams.update({"font.size": 24})

    hists = OrderedDict()
    for cut in cuts:
        sel, _ = utils.make_selection({cut_var: [cut, CUT_MAX_VAL]}, events_dict, bb_masks)
        h = utils.singleVarHist(
            events_dict, shape_var, bb_masks, weight_key=weight_key, selection=sel
        )

        hists[cut] = h[plot_key, ...] / np.sum(h[plot_key, ...].values())

        hep.histplot(
            hists[cut],
            yerr=True,
            label=f"{cut_var_label} >= {cut}",
            ax=ax,
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel(shape_var.label)
    ax.set_ylabel("Fraction of Events")
    ax.legend()

    if ratio:
        rax.hlines(1, shape_var.axis.edges[0], shape_var.axis.edges[-1], linestyle="--", alpha=0.5)
        vals_nocut = hists[0].values()

        next(rax._get_lines.prop_cycler)  # skip first
        for cut in cuts[1:]:
            hep.histplot(
                hists[cut] / vals_nocut,
                yerr=True,
                label=f"BDTScore >= {cut}",
                ax=rax,
                histtype="errorbar",
            )

        rax.set_ylim([0.4, 2.2])
        rax.set_ylabel("Ratio to Inclusive Shape")
        # rax.legend()

    if year == "all":
        hep.cms.label(
            "Preliminary",
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
        )
    else:
        hep.cms.label("Preliminary", data=True, lumi=f"{LUMI[year] / 1e3:.0f}", year=year, ax=ax)

    if in_ax:
        return

    if len(name):
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
