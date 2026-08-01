"""Print LaTeX tabular tables of load, generator, and branch parameters
from an EGRET-format case JSON (case14_gen_per_bus.json)."""

import json



def esc(name):
    """Escape LaTeX special characters in an element name."""
    return name.replace("_", r"\_")


def fmt(value):
    """Format a numeric parameter for a table cell."""
    if value is None:
        return "--"
    return f"{value:g}"


def latex_table(headers, rows, col_fmt, caption, label):
    """Assemble a LaTeX tabular (booktabs) from headers and string rows."""
    lines = [
        caption,
        label,
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def load_rows(loads):
    """Build sorted table rows for loads: name, bus, active/reactive demand."""
    rows = []
    for name, ld in sorted(loads.items(), key=lambda kv: int(kv[1]["bus"])):
        rows.append([esc(name), ld["bus"], fmt(ld["p_load"]), fmt(ld["q_load"])])
    return rows


def generator_rows(gens):
    """Build table rows for generators: bus, limits, and cost coefficients."""
    rows = []
    for name, g in sorted(gens.items(), key=lambda kv: int(kv[0])):
        c = g["p_cost"]["values"]
        rows.append([
            esc(name), g["bus"],
            fmt(g["p_min"]), fmt(g["p_max"]),
            fmt(c["0"]), fmt(c["1"]), fmt(c["2"]),
        ])
    return rows


def branch_rows(branches):
    """Build table rows for branches: endpoints, impedance, type, rating."""
    rows = []
    for name, b in sorted(branches.items(), key=lambda kv: int(kv[0])):
        rows.append([
            esc(name), b["from_bus"], b["to_bus"],
            fmt(b["resistance"]), fmt(b["reactance"]),
            fmt(b["charging_susceptance"]),
            b["branch_type"], fmt(b["rating_long_term"]),
        ])
    return rows


def main():
    path = "../data/case14_gen_per_bus.json"
    with open(path) as f:
        case = json.load(f)
    el = case["elements"]

    print(latex_table(
        ["Load", "Bus", r"$P_d$ (MW)", r"$Q_d$ (MVAr)"],
        load_rows(el["load"]),
        "llrr",
        "Load parameters for case14\\_gen\\_per\\_bus.",
        "tab:loads",
    ))
    print()
    print(latex_table(
        ["Gen", "Bus", r"$P_{\min}$", r"$P_{\max}$",
         r"$c_0$", r"$c_1$", r"$c_2$"],
        generator_rows(el["generator"]),
        "llrrrrr",
        "Generator limits (MW) and polynomial cost coefficients.",
        "tab:gens",
    ))
    print()
    print(latex_table(
        ["Branch", "From", "To", r"$R$ (pu)", r"$X$ (pu)", r"$B$ (pu)",
         "Type", "Rating (MVA)"],
        branch_rows(el["branch"]),
        "lllrrrlr",
        "Branch impedance and long-term thermal ratings.",
        "tab:branches",
    ))


if __name__ == "__main__":
    main()
