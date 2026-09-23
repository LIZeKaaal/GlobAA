"""
Plot the Section 3.3 nonlinear Helmholtz residual histories.

Input:
    results/nlh_main_history.csv

Output:
    figures/nlh_residual.pdf
"""

import argparse
import csv
import os

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join("/private/tmp", "global_anderson_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "results", "nlh_main_history.csv")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")
DEFAULT_OUTPUT_NAME = "nlh_residual.pdf"

M_GRID = ((1, 3, 5), (10, 20, 30))
X_AXIS_BY_ROW = (
    (0, 500, 100, 25.0),
    (0, 250, 50, 12.5),
)
Y_LIMITS = (5e-11, 3.0)
Y_MAJOR_TICKS = (1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
MARKER_RULE_BY_ROW = (
    (100, 40, 18),
    (50, 20, 8),
)
ALGORITHM_ORDER = (
    "GlobalAndersonSolver",
    "AA_pure",
    "FAA cs=0.1",
    "FAA cs=0.2",
    "Picard",
)
DISPLAY_LABELS = {
    "GlobalAndersonSolver": r"GlobAA($m$), $\eta=0$",
    "AA_pure": "pureAA($m$)",
    "FAA cs=0.1": "FAA($m$), cs=0.1",
    "FAA cs=0.2": "FAA($m$), cs=0.2",
}
STYLE = {
    "GlobalAndersonSolver": {
        "color": "#dc143c",
        "linestyle": "-",
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
    "FAA cs=0.1": {
        "color": "#2e8b57",
        "linestyle": (0, (7, 2.2)),
        "linewidth": 1.8,
    },
    "FAA cs=0.2": {
        "color": "#7b3294",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "x",
        "markersize": 7.2,
        "markeredgecolor": "#7b3294",
        "markeredgewidth": 1.45,
    },
    "Picard": {
        "color": "#1e90ff",
        "linestyle": "--",
        "linewidth": 1.8,
    },
}


def load_history(path):
    grouped = {}
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"algorithm", "m", "iter", "rel_res", "step_type"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"History CSV is missing columns: {missing_text}")

        for row in reader:
            key = (int(row["m"]), row["algorithm"])
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


def marker_indices_for_row(x_values, row_index):
    front_limit, front_step, back_step = MARKER_RULE_BY_ROW[row_index]
    marker_indices = []
    for idx, x_value in enumerate(x_values):
        x_int = int(round(x_value))
        step = front_step if x_int <= front_limit else back_step
        if x_int % step == 0:
            marker_indices.append(idx)

    if x_values and (len(x_values) - 1) not in marker_indices:
        last_x = int(round(x_values[-1]))
        step = front_step if last_x <= front_limit else back_step
        if not marker_indices:
            marker_indices.append(len(x_values) - 1)
        else:
            last_marked_x = int(round(x_values[marker_indices[-1]]))
            if last_x - last_marked_x >= 0.6 * step:
                marker_indices.append(len(x_values) - 1)
    return marker_indices


def plot_grid(history, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15.9, 10.4), sharey=True)
    legend_handles = {}

    for row_index, m_row in enumerate(M_GRID):
        for col_index, anderson_m in enumerate(m_row):
            ax = axes[row_index, col_index]
            for algorithm in ALGORITHM_ORDER:
                values = history.get((anderson_m, algorithm))
                if not values:
                    continue
                x_values = [item["iter"] for item in values]
                y_values = [item["rel_res"] for item in values]
                plot_style = dict(STYLE[algorithm])
                if algorithm in ("AA_pure", "FAA cs=0.2"):
                    plot_style["markevery"] = marker_indices_for_row(
                        x_values, row_index)
                line, = ax.semilogy(
                    x_values,
                    y_values,
                    label=algorithm,
                    **plot_style,
                )
                legend_handles.setdefault(algorithm, line)

            global_values = history.get((anderson_m, "GlobalAndersonSolver"))
            if global_values:
                mark_last_global_km_step(ax, global_values)

            ax.set_title(rf"$m={anderson_m}$", fontsize=18, pad=12)
            ax.set_xlabel(r"Iteration $k$", fontsize=18)
            x_start, x_stop, x_step, x_pad = X_AXIS_BY_ROW[row_index]
            ax.set_xlim(x_start - x_pad, x_stop + x_pad)
            ax.set_xticks(range(x_start, x_stop + x_step, x_step))
            if col_index == 0:
                ax.set_ylabel(r"$\Vert F_k\Vert/\Vert F_0\Vert$", fontsize=18)
            ax.set_ylim(*Y_LIMITS)
            ax.tick_params(axis="both", labelsize=12)
            ax.yaxis.set_major_locator(FixedLocator(Y_MAJOR_TICKS))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
            ax.grid(True, which="both", linestyle="-", linewidth=0.6,
                    alpha=0.35)

    handles = [
        legend_handles[algorithm]
        for algorithm in ALGORITHM_ORDER
        if algorithm in legend_handles
    ]
    labels = [
        DISPLAY_LABELS.get(algorithm, algorithm)
        for algorithm in ALGORITHM_ORDER
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
            handlelength=2.8,
            columnspacing=1.8,
            borderpad=0.45,
            bbox_to_anchor=(0.5, 0.985),
        )

    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.91))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the formal NLH residual grid.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    history = load_history(args.input)
    output_path = os.path.join(args.output_dir, args.output_name)
    plot_grid(history, output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
