"""Shared layout for the per-line tail grids: the two grid scaffolds and the axis styling.

Every per-line figure lays its panels out as one of two fixed 6-inch-wide (two-column) grids
and labels their y axes with one-significant-figure ticks.  Keeping the scaffolds and the
tick styling here lets a layout change land once for every grid rather than in each figure.

    panel_grid     one panel per line, a header legend and inch-based bands; the survivor,
                   box, violin and exact-distribution grids -- shared or independent axes
    margin_grid    a panel grid whose decoration margins are inches, not fractions, so the
                   figure grows downward with the row count -- the jittered strip grids
    fraction_grid  a panel grid whose margins and gaps are figure fractions, its height set
                   by the 0.95 panel aspect ratio -- the top-config and mu-difference grids
    one_sigfig_yaxis / fmt_one_sigfig   one-significant-figure y ticks for the small panels
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

FIG_W = 6.0                       # total figure width, inches (two-column text-block limit)
PANEL_ASPECT = 2.5 / 2.2          # margin_grid panel width / height
MARGIN_IN = {"left": 0.42, "right": 0.06, "bottom": 0.32, "top": 0.34}   # decoration bands
TITLE_IN, LEGEND_IN = 0.04, 0.19  # inches from the top edge down to the title / legend top
LABEL_IN = 0.03                   # inches from the bottom/left edge to the shared axis label
PANEL_TITLE_Y = 0.96             # panel_title anchor: text top just below the panel's top border
HEAD_IN, FOOT_IN = 0.55, 0.42     # panel_grid's fixed decoration bands: legend+title, x label
FONT = {"tick": 6, "panel_title": 7, "legend": 7, "suptitle": 7, "axlabel": 8}


def sci_label(v):
    """Mantissa-times-power-of-ten label with a minimal-digit mantissa, e.g. 1000 -> 1e3.

    The exponent is read off Python's own decimal formatting (not log10, which floors 1000
    to 1e2), so the mantissa is exact; `%g` then trims it (1000 -> 1e3, 1500 -> 1.5e3)."""
    mant_s, exp_s = f"{v:.6e}".split("e")
    return f"{float(mant_s):g}e{int(exp_s)}"


def fmt_one_sigfig(v, _pos):
    """One-sigfig tick label: scientific below 1e-3 (e.g. 1e-4), plain decimal at/above."""
    if v == 0:
        return "0"
    if abs(v) < 9.5e-3:
        mantissa, exponent = f"{v:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{v:g}"


def one_sigfig_yaxis(ax, nbins=4, steps=(1, 2, 10), sci_above=None):
    """Restrict y ticks to one-significant-figure values (leaves the y limits unchanged).

    `steps` are the allowed tick intervals (MaxNLocator convention).  The default (1, 2, 10)
    leaves no nice interval between 20 and 100, so a range a few hundred wide falls to
    20-spacing and overshoots nbins; add 5 for a 50-step on such ranges.  `sci_above` (when
    set) prints any tick at or above that magnitude in scientific form (1000 -> 1e3)."""
    def fmt(v, pos):
        if sci_above is not None and v != 0 and abs(v) >= sci_above:
            return sci_label(v)
        return fmt_one_sigfig(v, pos)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=list(steps)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def panel_title(ax, text, y=PANEL_TITLE_Y):
    """Per-line panel label placed inside the panel, just below its top border.

    Shared by every per-line grid so this placement lives in one spot.  `ax.text` in axes
    fractions with va='top' anchors the label's top edge reliably at the border, where
    `set_title(y=...)` (baseline-anchored) and `pad` do not."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT["panel_title"])


def margin_grid(nrow, ncol, sharey=False, aspect=PANEL_ASPECT, w=FIG_W, **kwargs):
    """Panel grid of fixed FIG_W width whose decoration margins are inches, not fractions.

    Panel width follows from the width left over after the margins; panel height follows
    from `aspect` (panel width / height), so the figure grows downward with the row count while
    the bands reserved for the suptitle, legend, and shared axis labels keep the same absolute
    size.  A larger `aspect` gives shorter panels, squeezing the vertical extent of tall grids."""
    panel_w = (w - MARGIN_IN["left"] - MARGIN_IN["right"]) / ncol
    h = nrow * panel_w / aspect + MARGIN_IN["top"] + MARGIN_IN["bottom"]
    fig, axs = plt.subplots(nrow, ncol, figsize=(w, h), squeeze=False, sharex=True, sharey=sharey)
    adjust = dict(left=MARGIN_IN["left"] / w, right=1 - MARGIN_IN["right"] / w,
                  bottom=MARGIN_IN["bottom"] / h, top=1 - MARGIN_IN["top"] / h,
                  wspace=0.24, hspace=0.12)
    adjust.update(kwargs)
    fig.subplots_adjust(**adjust)
    return fig, axs


def fraction_grid(nrow, ncol, head=0.07, foot=0.14, lft=0.1, rgt=0.14, 
                    wspace=0.020, hspace=0.04, sharex="col", sharey="row"):
    """Fixed-width nrow x ncol grid whose margins and gaps are figure fractions.

    head, foot, hspace are fractions of the figure height; lft, rgt, wspace of its width.
    The panel aspect is fixed at 0.95, so the figure height follows from the panel width, and
    rgt leaves a right band for a figure legend.  Returns (fig, axs, figh); the caller draws
    the panels and any legend.  Used by top_config_mu (one row) and mu_distribution_difference
    (two rows), which share this layout but pass their own margins."""
    xtick = FONT["tick"] / 72 
    pw_frac = (1 - lft - rgt - (ncol - 1) * wspace) / ncol
    panel_h = 0.95 * pw_frac * FIG_W
    ph_frac = (1 - head - foot - (nrow - 1) * hspace) / nrow
    figh = panel_h / ph_frac + xtick
    xtickf = xtick / figh
    fig, axs = plt.subplots(nrow, ncol, figsize=(FIG_W, figh), squeeze=False,
                            sharex=sharex, sharey=sharey)
    fig.subplots_adjust(left=lft, right=1 - rgt, bottom=foot + xtickf, top=1 - head,
                        wspace=wspace / pw_frac, hspace=hspace / ph_frac)
    return fig, axs, figh


def panel_grid(order_names, draw_panel, legend_handles, title, xlabel, ylabel, log_y=False,
               sharex=True, sharey=None, label=str, **layout):
    """Per-line panel grid: `fraction_grid` for the layout, the shared loop and decoration on
    top; returns (fig, axs).

    order_names     ordered line ids, one panel each (congested first)
    draw_panel      callback(ax, name) that draws one line's panel
    legend_handles  figure legend entries
    label           panel-title text from a line id (default `str`); the placement is shared
                    by `panel_title`, so panels never set their own title
    log_y           shared log y axis (survivor curves) vs independent linear (boxes)
    sharex          shared x axis (a common categorical axis) vs a per-line x range
    sharey          shared y axis; defaults to log_y (survivor curves share, boxes do not)
    **layout        margin fractions forwarded to `fraction_grid`; the defaults leave a left
                    band for the shared y label, and a sharey grid can override lft/wspace to
                    reclaim its dropped inner ticks.  The top/bottom bands are not forwarded --
                    they are pinned to a fixed inch size after (HEAD_IN, FOOT_IN), so the
                    legend and title band stays constant whatever the row count
    """
    if sharey is None:
        sharey = log_y
    ncol = min(5, len(order_names))
    nrow = min(4, int(np.ceil(len(order_names) / ncol)))
    opts = dict(head=0.24, foot=0.18, lft=0.10, rgt=0.02, wspace=0.05, hspace=0.06)
    opts.update(layout)
    fig, axs, figh = fraction_grid(nrow, ncol, sharex=sharex, sharey=sharey, **opts)
    if log_y:
        axs[0, 0].set_yscale("log")
    for name, ax in zip(order_names, axs.ravel()):
        draw_panel(ax, name)
        panel_title(ax, label(name))
        ax.tick_params(labelsize=5.5)
        if not sharex:                          # independent numeric x axes crowd at this size
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    for ax in axs.ravel()[len(order_names):]:
        ax.set_visible(False)

    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.26 / figh), ncol=len(legend_handles),
               frameon=True, fontsize=6, handlelength=1.6, borderpad=0.3)
    fig.suptitle(title, y=1 - 0.08 / figh, fontsize=8)
    fig.supxlabel(xlabel, y=0.07 / figh, fontsize=8)
    fig.supylabel(ylabel, x=0.06 / FIG_W, fontsize=8)
    fig.subplots_adjust(top=1 - HEAD_IN / figh, bottom=FOOT_IN / figh)   # fixed bands
    return fig, axs
