"""
Plot the three Section 3.1 counterexample figures from CSV histories.

Read the files produced by run_counterexample.py from --input-dir and write
the corresponding PDF figures to --output-dir.
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
    NullLocator,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")

CASES = (
    {
        "plot_kind": "full",
        "input_file": "counterexample_t0_2p1.csv",
        "output_file": "counterexample_t0_2p1.pdf",
        "scalar_orbit_iters": 20,
        "scalar_xticks": tuple(range(0, 26, 5)),
        "scalar_xlim": "half_main",
        "picard_marker_step": None,
    },
    {
        "plot_kind": "full",
        "input_file": "counterexample_t0_300.csv",
        "output_file": "counterexample_t0_300.pdf",
        "scalar_orbit_iters": 40,
        "scalar_xticks": tuple(range(0, 51, 10)),
        "scalar_xlim": "main",
        "picard_marker_step": 2,
    },
    {
        "plot_kind": "residual",
        "input_file": "counterexample_t0_2p1_m3_m5.csv",
        "output_file": "counterexample_t0_2p1_m3_m5_residual.pdf",
    },
)

ALGORITHM_ORDER = (
    "Picard",
    "AA_pure",
    "AA_residual",
    "GlobalAndersonSolver",
)
DISPLAY_LABELS = {
    "GlobalAndersonSolver": "GlobAA",
    "AA_pure": "pureAA",
    "AA_residual": "resAA",
}

AXIS_LABEL_FONTSIZE = 16
AXIS_TITLE_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
REFERENCE_LINE_COLOR = "#111111"
REFERENCE_TEXT_COLOR = "#111111"
REFERENCE_LINESTYLE = (0, (4.0, 2.2))
REFERENCE_LINEWIDTH = 1.6
STEP_AA_COLOR = "#d62728"
STEP_FALLBACK_COLOR = "#1f77b4"
RESIDUAL_ZERO_LINE = 2e-2
RESIDUAL_AXIS_BOTTOM_FACTOR = 0.96
RESIDUAL_CONVERGENCE_TOL = 1e-8
RESIDUAL_Y_TICKS = (1e0, 1e-1)
STEP_FALLBACK_MARKER_SIZE = 5.8
STEP_FALLBACK_SCATTER_SIZE = 28


def parse_float(value):
    if value == "":
        return math.nan
    return float(value)


def load_results(path):
    rows = []
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({
                "algorithm": row["algorithm"],
                "m": int(row["m"]),
                "iter": int(row["iter"]),
                "rel_res": float(row["rel_res"]),
                "residual": float(row["residual"]),
                "time": float(row["time"]),
                "step_type": row["step_type"],
                "diagnostic_residual": float(row["diagnostic_residual"]),
                "objective_value": float(row["objective_value"]),
                "x_norm": float(row["x_norm"]),
                "step_from_prev_norm": parse_float(
                    row["step_from_prev_norm"]),
                "s_value": float(row["s_value"]),
            })
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def group_results(rows):
    groups_by_key = {}
    for row in rows:
        key = (row["algorithm"], row["m"])
        groups_by_key.setdefault(key, []).append(row)

    def sort_key(item):
        algorithm, m_value = item[0]
        try:
            algorithm_index = ALGORITHM_ORDER.index(algorithm)
        except ValueError:
            algorithm_index = len(ALGORITHM_ORDER)
        return algorithm_index, m_value, algorithm

    groups = []
    for (algorithm, m_value), group_rows in sorted(groups_by_key.items(),
                                                  key=sort_key):
        group_rows.sort(key=lambda item: item["iter"])
        groups.append({
            "algorithm": algorithm,
            "m": m_value,
            "rows": group_rows,
        })
    return groups


def display_label(group):
    if group["algorithm"] in ("Picard", "Picard (GD)"):
        return "Picard"
    algorithm = DISPLAY_LABELS.get(
        group["algorithm"], group["algorithm"])
    return f"{algorithm}({group['m']})"


def generic_display_label(algorithm):
    if algorithm in ("Picard", "Picard (GD)"):
        return "Picard"
    return f"{DISPLAY_LABELS.get(algorithm, algorithm)}($m$)"


def order_legend_entries(handles, labels):
    prefixes = ("GlobAA", "pureAA", "resAA", "Picard")

    def sort_key(item):
        label = item[1]
        return next(
            (index for index, prefix in enumerate(prefixes)
             if label.startswith(prefix)),
            len(prefixes),
        )

    ordered = sorted(zip(handles, labels), key=sort_key)
    return tuple(zip(*ordered)) if ordered else ((), ())


def line_style_for_label(label):
    if label.startswith("Picard"):
        return ":"
    if label.startswith("AA_residual"):
        return "--"
    if label.startswith("AA_pure"):
        return "-."
    return "-"


def marker_for_label(label):
    if label.startswith("Picard"):
        return "o"
    if label.startswith("GlobalAndersonSolver"):
        return None
    if label.startswith("AA_residual"):
        return "o"
    if label.startswith("AA_pure"):
        return None
    return None


def color_for_label(label):
    if label.startswith("Picard"):
        return "tab:blue"
    if label.startswith("AA_pure"):
        return "tab:orange"
    if label.startswith("AA_residual"):
        return "tab:green"
    if label.startswith("GlobalAndersonSolver"):
        return "tab:red"
    return None


def zorder_for_label(label):
    if label.startswith("GlobalAndersonSolver"):
        return 6
    if label.startswith("Picard"):
        return 3
    return 4


def linewidth_for_label(label):
    if label.startswith("GlobalAndersonSolver"):
        return 3.3
    if label.startswith("Picard"):
        return 2.2
    return 2.2


def markerfacecolor_for_label(label):
    if label.startswith("Picard"):
        return "white"
    if label.startswith("AA_residual"):
        return color_for_label(label)
    return None


def markersize_for_label(label):
    if label.startswith("Picard"):
        return STEP_FALLBACK_MARKER_SIZE
    if label.startswith("GlobalAndersonSolver"):
        return 0
    if label.startswith("AA_residual"):
        return 6
    return 4.5


def markeredgewidth_for_label(label):
    if label.startswith("Picard"):
        return 1.2
    if label.startswith("AA_residual"):
        return 0.9
    return 1.4


def marker_indices_for_label(label, length, dense=False,
                             picard_marker_step=None):
    if marker_for_label(label) is None or length <= 0:
        return []
    if label.startswith("Picard"):
        if picard_marker_step is not None:
            step = max(1, int(picard_marker_step))
            indices = list(range(0, length, step))
            if indices[-1] != length - 1:
                indices.append(length - 1)
            return indices
        return list(range(length))
    if dense:
        return list(range(length))
    step = max(1, length // 10)
    indices = list(range(0, length, step))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def line_markevery_for_label(label, length, dense=False):
    if label.startswith("Picard"):
        return []
    indices = marker_indices_for_label(label, length, dense=dense)
    return indices if indices else None


def draw_picard_marker(ax, x_values, y_values, label, dense=False,
                       picard_marker_step=None):
    if not label.startswith("Picard"):
        return
    indices = marker_indices_for_label(
        label, len(y_values), dense=dense,
        picard_marker_step=picard_marker_step)
    if not indices:
        return
    ax.plot(
        [x_values[index] for index in indices],
        [y_values[index] for index in indices],
        linestyle="None",
        marker=marker_for_label(label),
        markersize=markersize_for_label(label),
        color=color_for_label(label),
        markerfacecolor=markerfacecolor_for_label(label),
        markeredgewidth=markeredgewidth_for_label(label),
        zorder=8,
        label="_nolegend_",
    )


def positive_plot_floor(value_groups):
    positives = []
    for values in value_groups:
        positives.extend(value for value in values
                         if math.isfinite(value) and value > 0)
    if not positives:
        return 1e-16
    return max(min(positives) * 0.1, 1e-16)


def add_log_zero_reference(ax, floor):
    if floor <= 0:
        return
    ax.axhline(floor, color=REFERENCE_LINE_COLOR,
               linewidth=REFERENCE_LINEWIDTH,
               linestyle=REFERENCE_LINESTYLE, alpha=0.98, zorder=2)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(floor * 0.55, 1e-300), y_max)
    x_min, x_max = ax.get_xlim()
    label_x = x_max - 0.04 * (x_max - x_min)
    ax.text(label_x, floor, r"$0$", ha="right", va="center",
            color=REFERENCE_TEXT_COLOR, fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            })


def configure_relative_residual_axis(ax, residual_value_groups):
    finite_values = [
        value
        for values in residual_value_groups
        for value in values
        if math.isfinite(value) and value > 0
    ]
    y_upper = max(1.25, max(finite_values) * 1.16 if finite_values else 1.25)
    ax.set_ylim(RESIDUAL_ZERO_LINE * RESIDUAL_AXIS_BOTTOM_FACTOR, y_upper)
    ax.yaxis.set_major_locator(FixedLocator(RESIDUAL_Y_TICKS))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())


def relative_residual_plot_series(rows):
    x_values = []
    residuals = []
    converged_seen = False
    for row in rows:
        value = row["rel_res"]
        if not math.isfinite(value):
            continue
        if value <= RESIDUAL_CONVERGENCE_TOL:
            if converged_seen:
                continue
            plot_value = RESIDUAL_ZERO_LINE
            converged_seen = True
        elif value < RESIDUAL_ZERO_LINE:
            continue
        else:
            plot_value = value
        x_values.append(row["iter"])
        residuals.append(plot_value)
    return x_values, residuals


def find_group(groups, algorithm):
    for group in groups:
        if group["algorithm"] == algorithm:
            return group
    return None


def find_group_for_m(groups, algorithm, m_value):
    for group in groups:
        if group["algorithm"] == algorithm and group["m"] == m_value:
            return group
    return None


def draw_single_step_type_strip(ax, group, reference_xlim, title,
                                show_legend=False, display_title=None):
    aa_x = []
    fallback_x = []
    if group is not None:
        rows = group["rows"]
        for previous_row, current_row in zip(rows[:-1], rows[1:]):
            x_value = previous_row["iter"]
            if current_row["step_type"] == "AA":
                aa_x.append(x_value)
            elif current_row["step_type"] != "initial":
                fallback_x.append(x_value)

    if aa_x:
        ax.scatter(
            aa_x,
            [0.0] * len(aa_x),
            marker="*",
            s=56,
            color=STEP_AA_COLOR,
            edgecolors=STEP_AA_COLOR,
            linewidths=0.7,
            zorder=4,
        )
    if fallback_x:
        ax.scatter(
            fallback_x,
            [0.0] * len(fallback_x),
            marker="o",
            s=30,
            facecolors="white",
            edgecolors=STEP_FALLBACK_COLOR,
            linewidths=1.2,
            zorder=5,
        )

    ax.set_xlim(reference_xlim)
    ax.set_ylim(-0.52, 0.52)
    ax.set_yticks([])
    ax.set_xlabel(display_title if display_title is not None else title,
                  fontsize=13)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(True, axis="x", which="major", linestyle="--",
            linewidth=0.7, alpha=0.35)
    ax.grid(False, axis="y")

    if show_legend:
        legend_handles = [
            plt.Line2D([0], [0], marker="*", linestyle="None",
                       markersize=8, markerfacecolor=STEP_AA_COLOR,
                       markeredgecolor=STEP_AA_COLOR, label="AA step"),
            plt.Line2D([0], [0], marker="o", linestyle="None",
                       markersize=6, markerfacecolor="white",
                       markeredgecolor=STEP_FALLBACK_COLOR,
                       markeredgewidth=1.2, label="Picard/KM step"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=1,
                  frameon=True, fontsize=8, handletextpad=0.45,
                  labelspacing=0.3, borderpad=0.3)


def step_type_x_positions(group):
    aa_x = []
    fallback_x = []
    if group is not None:
        rows = group["rows"]
        for previous_row, current_row in zip(rows[:-1], rows[1:]):
            x_value = previous_row["iter"]
            if current_row["step_type"] == "AA":
                aa_x.append(x_value)
            elif current_row["step_type"] != "initial":
                fallback_x.append(x_value)
    return aa_x, fallback_x


def draw_step_type_axis_markers(ax, group):
    aa_x, fallback_x = step_type_x_positions(group)
    x_min, x_max = ax.get_xlim()
    aa_x = [x_value for x_value in aa_x if x_min <= x_value <= x_max]
    fallback_x = [
        x_value for x_value in fallback_x if x_min <= x_value <= x_max
    ]
    transform = ax.get_xaxis_transform()
    if fallback_x:
        ax.scatter(
            fallback_x,
            [0.0] * len(fallback_x),
            transform=transform,
            marker="o",
            s=STEP_FALLBACK_SCATTER_SIZE,
            facecolors="white",
            edgecolors=STEP_FALLBACK_COLOR,
            linewidths=1.2,
            zorder=30,
            clip_on=False,
        )
    if aa_x:
        ax.scatter(
            aa_x,
            [0.0] * len(aa_x),
            transform=transform,
            marker="*",
            s=42,
            color=STEP_AA_COLOR,
            edgecolors=STEP_AA_COLOR,
            linewidths=0.7,
            zorder=31,
            clip_on=False,
        )


def step_legend_handles():
    return [
        plt.Line2D([0], [0], marker="*", linestyle="None",
                   markersize=7, markerfacecolor=STEP_AA_COLOR,
                   markeredgecolor=STEP_AA_COLOR, label="AA step"),
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markersize=STEP_FALLBACK_MARKER_SIZE,
                   markerfacecolor="white",
                   markeredgecolor=STEP_FALLBACK_COLOR,
                   markeredgewidth=1.2, label="Picard/KM step"),
    ]


def plot_residual_sweep(groups, output_path):
    m_values = sorted({group["m"] for group in groups})
    fig, axes = plt.subplots(
        1, len(m_values), figsize=(12, 5.1))
    if len(m_values) == 1:
        axes = [axes]

    x_max = max(
        row["iter"] for group in groups for row in group["rows"])
    x_limits = (-0.03 * x_max, 1.03 * x_max)
    for ax, m_value in zip(axes, m_values):
        residual_value_groups = []
        for group in groups:
            if group["m"] != m_value:
                continue
            label = group["algorithm"]
            rows = group["rows"]
            x_values, residuals = relative_residual_plot_series(rows)
            residual_value_groups.append(residuals)
            ax.semilogy(
                x_values,
                residuals,
                linestyle=line_style_for_label(label),
                marker=marker_for_label(label),
                markersize=markersize_for_label(label),
                markevery=line_markevery_for_label(label, len(residuals)),
                color=color_for_label(label),
                markerfacecolor=markerfacecolor_for_label(label),
                markeredgewidth=markeredgewidth_for_label(label),
                linewidth=linewidth_for_label(label),
                solid_capstyle="butt",
                dash_capstyle="butt",
                zorder=zorder_for_label(label),
                label=generic_display_label(label),
            )
            draw_picard_marker(ax, x_values, residuals, label)

        ax.set_xlim(x_limits)
        ax.set_xlabel(r"Iteration $k$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$\|F_k\|/\|F_0\|$",
                      fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(fr"$m={m_value}$", fontsize=AXIS_TITLE_FONTSIZE)
        configure_relative_residual_axis(ax, residual_value_groups)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.grid(False, which="minor")
        draw_step_type_axis_markers(
            ax, find_group_for_m(groups, "GlobalAndersonSolver", m_value))

    handles, labels = order_legend_entries(
        *axes[0].get_legend_handles_labels())
    step_handles = step_legend_handles()
    handles = tuple(handles) + tuple(step_handles)
    labels = tuple(labels) + tuple(handle.get_label()
                                   for handle in step_handles)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
               frameon=True,
               bbox_to_anchor=(0.02, 0.96, 0.96, 0.04),
               fontsize=LEGEND_FONTSIZE, handlelength=2.8,
               handletextpad=0.75, columnspacing=1.15,
               markerscale=1.25, borderpad=0.45)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_groups(groups, output_path, scalar_orbit_iters, scalar_xticks,
                scalar_xlim, picard_marker_step):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    residual_ax, scalar_ax = axes
    main_axes = (residual_ax, scalar_ax)

    residual_value_groups = []
    for group in groups:
        algorithm = group["algorithm"]
        label = display_label(group)
        rows = group["rows"]
        x_values = [row["iter"] for row in rows]
        residual_x, residuals = relative_residual_plot_series(rows)
        residual_value_groups.append(residuals)
        s_values = [row["s_value"] for row in rows]

        style = {
            "linestyle": line_style_for_label(algorithm),
            "marker": marker_for_label(algorithm),
            "markersize": markersize_for_label(algorithm),
            "color": color_for_label(algorithm),
            "markerfacecolor": markerfacecolor_for_label(algorithm),
            "markeredgewidth": markeredgewidth_for_label(algorithm),
            "linewidth": linewidth_for_label(algorithm),
            "solid_capstyle": "butt",
            "dash_capstyle": "butt",
            "zorder": zorder_for_label(algorithm),
            "label": label,
        }

        residual_ax.semilogy(
            residual_x,
            residuals,
            markevery=line_markevery_for_label(algorithm, len(residuals)),
            **style,
        )
        draw_picard_marker(
            residual_ax, residual_x, residuals, algorithm,
            picard_marker_step=picard_marker_step)

        window_stop = min(len(s_values), scalar_orbit_iters + 1)
        scalar_x = x_values[:window_stop]
        scalar_y = s_values[:window_stop]
        scalar_ax.plot(
            scalar_x,
            scalar_y,
            markevery=line_markevery_for_label(algorithm, len(scalar_y),
                                               dense=True),
            **style,
        )
        draw_picard_marker(
            scalar_ax, scalar_x, scalar_y, algorithm, dense=True,
            picard_marker_step=picard_marker_step)

    x_label = r"Iteration $k$"
    residual_ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    residual_ax.set_ylabel(r"$\|F_k\|/\|F_0\|$",
                           fontsize=AXIS_LABEL_FONTSIZE)

    scalar_ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    scalar_ax.set_ylabel(r"$s(u_k)$", fontsize=AXIS_LABEL_FONTSIZE)

    configure_relative_residual_axis(residual_ax, residual_value_groups)

    orbit = 249.0 * (math.sqrt(5.0) - 2.0)
    reference_values = (-249.0, -orbit, orbit, 249.0)
    for value in reference_values:
        scalar_ax.axhline(value, color=REFERENCE_LINE_COLOR,
                          linewidth=REFERENCE_LINEWIDTH,
                          linestyle=REFERENCE_LINESTYLE, alpha=0.98,
                          zorder=2)

    x_min, x_max = scalar_ax.get_xlim()
    y_min, y_max = scalar_ax.get_ylim()
    y_span = y_max - y_min
    residual_x_min, residual_x_max = residual_ax.get_xlim()
    if scalar_xlim == "main":
        axis_left = residual_x_min
        axis_right = residual_x_max
    else:
        axis_left = residual_x_min / 2.0
        axis_right = residual_x_max / 2.0
    scalar_ax.set_xlim(axis_left, axis_right)
    scalar_ax.set_xticks(scalar_xticks)
    scalar_ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)
    label_x = axis_right - 0.04 * (axis_right - axis_left)
    label_y_offset = 0.012 * y_span
    for value in reference_values:
        scalar_ax.text(label_x, value - label_y_offset, f"{value:.2f}",
                       ha="right", va="top", color=REFERENCE_TEXT_COLOR,
                       fontsize=10, zorder=9,
                       bbox={
                           "boxstyle": "round,pad=0.15",
                           "facecolor": "white",
                           "edgecolor": "none",
                           "alpha": 0.75,
                       })

    for ax in main_axes:
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.grid(False, which="minor")

    draw_step_type_axis_markers(
        residual_ax, find_group(groups, "GlobalAndersonSolver"))

    handles, labels = order_legend_entries(
        *residual_ax.get_legend_handles_labels())
    step_handles = [
        plt.Line2D([0], [0], marker="*", linestyle="None",
                   markersize=7, markerfacecolor=STEP_AA_COLOR,
                   markeredgecolor=STEP_AA_COLOR, label="AA step"),
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markersize=STEP_FALLBACK_MARKER_SIZE,
                   markerfacecolor="white",
                   markeredgecolor=STEP_FALLBACK_COLOR,
                   markeredgewidth=1.2, label="Picard/KM step"),
    ]
    handles = tuple(handles) + tuple(step_handles)
    labels = tuple(labels) + tuple(handle.get_label()
                                   for handle in step_handles)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
               frameon=True,
               bbox_to_anchor=(0.02, 0.96, 0.96, 0.04),
               fontsize=LEGEND_FONTSIZE, handlelength=2.8,
               handletextpad=0.75, columnspacing=1.15,
               markerscale=1.25, borderpad=0.45)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the Section 3.1 counterexample figure from CSV.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    for case in CASES:
        input_path = os.path.join(args.input_dir, case["input_file"])
        output_path = os.path.join(args.output_dir, case["output_file"])
        rows = load_results(input_path)
        groups = group_results(rows)
        if case["plot_kind"] == "full":
            plot_groups(groups, output_path, case["scalar_orbit_iters"],
                        case["scalar_xticks"], case["scalar_xlim"],
                        case["picard_marker_step"])
        elif case["plot_kind"] == "residual":
            plot_residual_sweep(groups, output_path)
        else:
            raise ValueError(f"unknown plot kind: {case['plot_kind']}")
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
