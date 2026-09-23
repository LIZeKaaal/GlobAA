"""
Plot the Section 3.2 ENR/ISTA residual histories.

Input:
    results/enr_ista_history.csv

Output:
    figures/enr_ista_madelon_residual.pdf
    figures/enr_ista_ill_conditioned_residual.pdf
"""

import argparse
import csv
import math
import os

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join("/private/tmp", "global_anderson_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FixedLocator,
    LogFormatterMathtext,
    LogLocator,
    ScalarFormatter,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "results", "enr_ista_history.csv")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")

CASE_OUTPUTS = {
    "contractive_madelon": "enr_ista_madelon_residual.pdf",
    "nonexpansive_ill_conditioned": "enr_ista_ill_conditioned_residual.pdf",
}

M_VALUES = (10, 20, 30)
AXIS_BOX_ASPECT = 0.75
X_AXIS_BY_CASE = {
    "contractive_madelon": (0, 2000, 500, 100.0),
    "nonexpansive_ill_conditioned": (0, 25000, 5000, 1250.0),
}
Y_LIMITS = (5e-11, 3.0)
Y_MAJOR_TICKS = (1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
GLOBAL_ALGORITHM = "GlobalAndersonSolver eta=0"
REGULARIZED_GLOBAL_BY_CASE = {
    "contractive_madelon": "GlobalAndersonSolver eta=1e-08",
    "nonexpansive_ill_conditioned": "GlobalAndersonSolver eta=1e-10",
}
DISPLAY_LABELS = {
    GLOBAL_ALGORITHM: r"GlobAA($m$), $\eta=0$",
    "GlobalAndersonSolver eta=1e-08": r"GlobAA($m$), $\eta=10^{-8}$",
    "GlobalAndersonSolver eta=1e-10": r"GlobAA($m$), $\eta=10^{-10}$",
    "AA_pure": "pureAA($m$)",
    "AA_residual": "resAA($m$)",
    "KM lambda=0.5": r"KM, $\lambda=0.5$",
}
STYLE = {
    GLOBAL_ALGORITHM: {
        "color": "#dc143c",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "GlobalAndersonSolver eta=1e-08": {
        "color": "#dc143c",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "GlobalAndersonSolver eta=1e-10": {
        "color": "#dc143c",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "AA_pure": {
        "color": "#ff7f0e",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "^",
        "markersize": 6.8,
        "markerfacecolor": "#ff7f0e",
        "markeredgecolor": "#ff7f0e",
        "markeredgewidth": 1.25,
    },
    "AA_residual": {
        "color": "#2e8b57",
        "linestyle": (0, (7, 2.2)),
        "linewidth": 1.8,
    },
    "Picard": {
        "color": "#1e90ff",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    "KM lambda=0.5": {
        "color": "#1e90ff",
        "linestyle": "-",
        "linewidth": 1.8,
    },
}


def algorithm_order(case_name):
    return (
        GLOBAL_ALGORITHM,
        REGULARIZED_GLOBAL_BY_CASE[case_name],
        "AA_pure",
        "AA_residual",
        "Picard",
        "KM lambda=0.5",
    )


def load_history(path):
    grouped = {}
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "case",
            "algorithm",
            "m",
            "iter",
            "rel_res",
            "step_type",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"History CSV is missing columns: {missing_text}")

        for row in reader:
            key = (row["case"], int(row["m"]), row["algorithm"])
            grouped.setdefault(key, []).append({
                "iter": int(row["iter"]),
                "rel_res": max(float(row["rel_res"]), 1e-300),
                "step_type": row["step_type"],
            })

    for key, values in grouped.items():
        grouped[key] = sorted(values, key=lambda item: item["iter"])
    return grouped


def mark_last_global_km_step(ax, values):
    km_values = [item for item in values if item["step_type"] == "KM"]
    if not km_values:
        return

    last_km = km_values[-1]
    ax.axvline(
        last_km["iter"],
        color="#dc143c",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
    )
    ax.semilogy(
        [last_km["iter"]],
        [last_km["rel_res"]],
        marker="*",
        markersize=12,
        markerfacecolor="#dc143c",
        markeredgecolor="#8f2f33",
        markeredgewidth=0.6,
        linestyle="None",
        zorder=5,
    )


def marker_indices_for_values(x_values, target_count=9):
    if not x_values:
        return []
    if len(x_values) <= target_count:
        return list(range(len(x_values)))

    step = max(1, int(math.ceil(len(x_values) / target_count)))
    marker_indices = list(range(0, len(x_values), step))
    if marker_indices[-1] != len(x_values) - 1:
        marker_indices.append(len(x_values) - 1)
    return marker_indices


def nice_tick_step(max_x):
    if max_x <= 10:
        return 1

    raw_step = max_x / 5.0
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if raw_step <= step:
            return int(step)
    return int(10 * magnitude)


def case_x_axis(history, case_name):
    if case_name in X_AXIS_BY_CASE:
        return X_AXIS_BY_CASE[case_name]

    max_iter = 0
    for (row_case, anderson_m, _algorithm), values in history.items():
        if row_case != case_name or anderson_m not in M_VALUES or not values:
            continue
        max_iter = max(max_iter, values[-1]["iter"])

    x_stop = max(1, max_iter)
    x_step = nice_tick_step(x_stop)
    x_stop = int(math.ceil(x_stop / x_step) * x_step)
    x_pad = max(0.04 * x_stop, 0.5)
    return 0, x_stop, x_step, x_pad


def plot_case(history, case_name, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.9, 5.9), sharey=True)
    legend_handles = {}
    x_start, x_stop, x_step, x_pad = case_x_axis(history, case_name)
    plotted_algorithms = algorithm_order(case_name)

    for col_index, anderson_m in enumerate(M_VALUES):
        ax = axes[col_index]

        for algorithm in plotted_algorithms:
            values = history.get((case_name, anderson_m, algorithm))
            if not values:
                continue
            x_values = [item["iter"] for item in values]
            plot_style = dict(STYLE[algorithm])
            if algorithm == "AA_pure":
                plot_style["markevery"] = marker_indices_for_values(x_values)
            line, = ax.semilogy(
                x_values,
                [item["rel_res"] for item in values],
                label=algorithm,
                **plot_style,
            )
            legend_handles.setdefault(algorithm, line)

        global_values = history.get(
            (case_name, anderson_m, GLOBAL_ALGORITHM))
        if global_values:
            mark_last_global_km_step(ax, global_values)

        regularized_global_values = history.get(
            (case_name, anderson_m, REGULARIZED_GLOBAL_BY_CASE[case_name]))
        if regularized_global_values:
            mark_last_global_km_step(ax, regularized_global_values)

        ax.set_title(rf"$m={anderson_m}$", fontsize=18, pad=12)
        ax.set_xlabel(r"Iteration $k$", fontsize=18)
        ax.set_xlim(x_start - x_pad, x_stop + x_pad)
        ax.set_xticks(range(x_start, x_stop + x_step, x_step))
        if x_stop >= 10000:
            x_formatter = ScalarFormatter(useMathText=True)
            x_formatter.set_scientific(True)
            x_formatter.set_powerlimits((4, 4))
            ax.xaxis.set_major_formatter(x_formatter)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="both", linestyle="-", linewidth=0.6,
                alpha=0.35)
        ax.set_ylim(*Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(Y_MAJOR_TICKS))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

        if col_index == 0:
            ax.set_ylabel(r"$\Vert F_k\Vert/\Vert F_0\Vert$", fontsize=18)
        ax.set_box_aspect(AXIS_BOX_ASPECT)

    handles = [
        legend_handles[algorithm]
        for algorithm in plotted_algorithms
        if algorithm in legend_handles
    ]
    labels = [
        DISPLAY_LABELS.get(algorithm, algorithm)
        for algorithm in plotted_algorithms
        if algorithm in legend_handles
    ]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            fontsize=15,
            handlelength=3.1,
            columnspacing=2.0,
            borderpad=0.5,
            bbox_to_anchor=(0.5, 0.94),
        )

    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.91))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot formal ENR/ISTA residual grids.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    history = load_history(args.input)

    for case_name, output_name in CASE_OUTPUTS.items():
        has_case = any(key[0] == case_name for key in history)
        if not has_case:
            print(f"Skipping {case_name}: no rows found in {args.input}")
            continue
        output_path = os.path.join(args.output_dir, output_name)
        plot_case(history, case_name, output_path)
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
