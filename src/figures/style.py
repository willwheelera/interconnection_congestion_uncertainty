"""Shared plotting parameters and helpers for the sampler-comparison figures.

Sampler colors and labels, the samplers the figures draw by default, and the standard
median + 10-90 replicate band, so every plot in the package reads samplers with one
consistent color, name, and band style.

Selecting samplers.  Every comparison figure takes one option, `--methods`, whose values
are the method names as they appear in the data:

    (default)                        direct and ascent-IS at beta 0.2
    --methods all                    everything the sweep wrote
    --methods direct,ascent_is       ascent-IS at every beta it was run at
    --methods ascent_is_b0.5         one beta on its own
    --methods direct,subset_sim,smc_tilt,cross_entropy

A name is either a full method key (`ascent_is_b0.5`), a sampler family that covers each
of its betas (`ascent_is`), or `all`.  A name matching nothing in the data is an error,
so a typo is visible instead of silently dropping a curve.  Anything but the default is
tagged into the output filename, so a restricted figure never overwrites the default one.
"""

import numpy as np
from matplotlib import colormaps

COLORS = {
    "direct": "tab:red", "subset_sim": "tab:blue", "ascent_is": "tab:green",
    "smc_tilt": "tab:purple", "ascent_exact": "tab:orange", "cross_entropy": "tab:brown",
}
_VIRIDIS = colormaps["viridis"](np.linspace(0.1, 0.8, 4))   # dark to light, one per law
DIST_COLOR = {                      # law -> color, shared by the per-line enumeration figures
    "bernoulli": (_VIRIDIS[0] + np.array([.2, 0, 0, 0])), "mrf_a": _VIRIDIS[1],
    "mrf_cobuild": _VIRIDIS[2], "mrf_substitute": _VIRIDIS[3],
}
LABELS = {
    "direct": "direct",
    "subset_sim": "subset sim",
    "ascent_is": "ascent-IS",
    "smc_tilt": "sequential tilt",
    "ascent_exact": "ascent-IS+exact",
    "cross_entropy": "cross-entropy",
}
BETA_COLORS = {                      # ascent-IS by defensive weight: one green ramp,
    "0.2": "#1a9850", "0.5": "#66bd63", "0.8": "#a6d96a",   # kept off black (the truth)
}
METHOD_ORDER = ["direct", "subset_sim", "smc_tilt", "cross_entropy",
                "ascent_is", "ascent_exact"]
DEFAULT_METHODS = ("direct", "ascent_is_b0.2")   # what every comparison figure draws
MOVE_STYLE = {"na": "-", "all": "-", "best": ":", "top5": "--"}


def budget_colors(budgets):
    """One color per sampling budget along a light-to-dark ramp, kept off the gray exact fill."""
    shades = colormaps["plasma"](np.linspace(0.1, 0.65, len(budgets)))
    return {b: shades[k] for k, b in enumerate(budgets)}


def parse_method(method):
    """(base sampler, beta string or None) for a method key such as `ascent_is_b0.5`."""
    base, _, beta = method.partition("_b")
    return (base, beta) if beta else (method, None)


def method_color(method):
    """Plot color of a method key; the ascent-IS betas share a green ramp."""
    base, beta = parse_method(method)
    if beta is not None:
        return BETA_COLORS.get(beta, COLORS.get(base, "gray"))
    return COLORS.get(base, "gray")


def method_label(method):
    """Legend label of a method key, carrying its beta when it has one."""
    base, beta = parse_method(method)
    label = LABELS.get(base, base)
    return label if beta is None else rf"{label} ($\beta$={beta})"


def series_label(method, move):
    """Method label, annotated with the ascent move when it is not the default."""
    return method_label(method) + ("" if move == "na" else f" ({move})")


def select_methods(present, methods=DEFAULT_METHODS):
    """The method keys of `present` that `methods` asks for, in plotting order.

    Each requested name is a full key, a sampler family covering its betas, or `all`;
    a name that matches nothing present raises.
    """
    keys = [m for m in present if m != "truth"]
    if tuple(methods) == ("all",):
        keep = set(keys)
    else:
        keep = set()
        for name in methods:
            matched = {m for m in keys if m == name or parse_method(m)[0] == name}
            if not matched:
                raise ValueError(f"no method {name!r} in the data; present: {sorted(keys)}")
            keep |= matched
    rank = {b: i for i, b in enumerate(METHOD_ORDER)}
    return sorted(keep, key=lambda m: (rank.get(parse_method(m)[0], len(rank)),
                                       float(parse_method(m)[1] or 0.0)))


def selection_label(methods=DEFAULT_METHODS):
    """Title fragment naming the selection, e.g. 'direct, ascent-IS (beta=0.2)'."""
    if tuple(methods) == ("all",):
        return "all methods"
    return ", ".join(method_label(m) for m in methods)


def series_tag(methods=DEFAULT_METHODS):
    """Filename fragment naming a non-default selection ('' for the default)."""
    return "" if tuple(methods) == DEFAULT_METHODS else "_" + "-".join(methods)


def add_method_args(parser):
    """The one sampler-selection option every comparison figure takes."""
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="samplers to draw, comma separated: method keys "
                             "(direct, ascent_is_b0.5), a family covering its betas "
                             "(ascent_is), or all")


def method_selection(args):
    """The requested method names."""
    return tuple(args.methods.split(","))


def band(ax, b, est, color, label):
    """Median curve with a 10-90 percentile band over replicates."""
    med = np.median(est, axis=0)
    lo, hi = np.percentile(est, [10, 90], axis=0)
    ax.plot(b, med, color=color, lw=2, label=label)
    ax.fill_between(b, lo, hi, color=color, alpha=0.18)
