"""
Plot the Section 3.4 Navier-Stokes residual histories and velocity fields.

Read the m=50 runs from results_10, results_50, and results_100, including their
saved mesh configurations and final velocity states. Write
figures/ns_report_basepoints_residual.pdf and
figures/ns_report_basepoints_velocity.pdf without rerunning the solvers.
"""

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join("/private/tmp", "global_anderson_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator
import numpy as np

from run_NS import TaylorHoodCavityProblem


RESULT_SPECS = (
    ("results_10", os.path.join(SCRIPT_DIR, "results_10")),
    ("results_50", os.path.join(SCRIPT_DIR, "results_50")),
    ("results_100", os.path.join(SCRIPT_DIR, "results_100")),
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")
RESIDUAL_OUTPUT = "ns_report_basepoints_residual.pdf"
VELOCITY_OUTPUT = "ns_report_basepoints_velocity.pdf"
ANDERSON_M = 50
GLOBAL_KEY = "global_eta0"
GLOBAL_FINAL_STATE_KEY = f"m{ANDERSON_M}_{GLOBAL_KEY}"

DISPLAY_LABELS = {
    "GlobalAndersonSolver": r"GlobAA($m$), $\eta=0$",
    "AA_pure": "pureAA($m$)",
    "FAA cs=0.2": "FAA($m$), cs=0.2",
    "FAA cs=1/sqrt(2)": "FAA($m$), cs=1/sqrt(2)",
}


def display_label_for_m(algorithm, m_value):
    return DISPLAY_LABELS.get(algorithm, algorithm).replace(
        "($m$)", f"({m_value})")

ALGORITHM_ORDER = (
    "global_eta0",
    "aa_pure",
    "faa02",
    "faa0707",
    "picard",
)
Y_LIMITS = (5e-11, 3.0)
Y_MAJOR_TICKS = (1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
MARKER_RULE = (50, 20, 8)
VELOCITY_AXIS_LIMITS = (-0.05, 1.05)
VELOCITY_AXIS_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
LEGEND_ORDER = (
    "GlobalAndersonSolver",
    "AA_pure",
    "FAA cs=0.2",
    "FAA cs=1/sqrt(2)",
    "Picard",
)
STYLE = {
    "Picard": {
        "color": "#1e90ff",
        "linestyle": "--",
        "linewidth": 1.8,
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
    "GlobalAndersonSolver": {
        "color": "#dc143c",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "FAA cs=0.2": {
        "color": "#2e8b57",
        "linestyle": (0, (7, 2.2)),
        "linewidth": 1.8,
    },
    "FAA cs=1/sqrt(2)": {
        "color": "#7b3294",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "x",
        "markersize": 7.2,
        "markeredgecolor": "#7b3294",
        "markeredgewidth": 1.45,
    },
}


def load_config(results_dir):
    with open(os.path.join(results_dir, "ns_run_config.json")) as json_file:
        return json.load(json_file)


def load_history(results_dir):
    grouped = {}
    path = os.path.join(results_dir, "ns_main_history.csv")
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"method_key", "algorithm", "m", "iter", "rel_res",
                    "step_type"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {missing_text}")

        for row in reader:
            if int(row["m"]) != ANDERSON_M:
                continue
            key = row["method_key"]
            grouped.setdefault(key, []).append({
                "algorithm": row["algorithm"],
                "iter": int(row["iter"]),
                "rel_res": max(float(row["rel_res"]), 1e-300),
                "step_type": row["step_type"],
            })

    for key, values in grouped.items():
        grouped[key] = sorted(values, key=lambda item: item["iter"])
    return grouped


def result_title(config):
    dofs = int(config["mixed_dofs"])
    reynolds = float(config["re"])
    return rf"DOFs={dofs:,}, $Re={reynolds:g}$"


def residual_title(config):
    return result_title(config)


def build_problem(config):
    return TaylorHoodCavityProblem(
        base_points=int(config["base_points"]),
        reynolds=float(config["re"]),
        grad_div=float(config["grad_div"]),
        edge_width=float(config["edge_width"]),
        refine_edges=bool(config["refine_edges"]),
        quadrature_order=int(config["quadrature_order"]),
    )


def marker_indices(x_values):
    front_limit, front_step, back_step = MARKER_RULE
    marker_positions = []
    for index, x_value in enumerate(x_values):
        x_int = int(round(x_value))
        step = front_step if x_int <= front_limit else back_step
        if x_int % step == 0:
            marker_positions.append(index)

    if x_values and (len(x_values) - 1) not in marker_positions:
        last_x = int(round(x_values[-1]))
        step = front_step if last_x <= front_limit else back_step
        if not marker_positions:
            marker_positions.append(len(x_values) - 1)
        else:
            last_marked_x = int(round(x_values[marker_positions[-1]]))
            if last_x - last_marked_x >= 0.6 * step:
                marker_positions.append(len(x_values) - 1)
    return marker_positions


def mark_last_km_step(ax, values, style):
    km_values = [item for item in values if item["step_type"] == "KM"]
    if not km_values:
        return
    last_km = km_values[-1]
    color = style.get("color", "#dc143c")
    ax.axvline(
        last_km["iter"],
        color=color,
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
        markerfacecolor=color,
        markeredgecolor="#8f2f33",
        markeredgewidth=0.6,
        linestyle="None",
        zorder=5,
    )


def load_result_bundle():
    bundles = []
    for name, results_dir in RESULT_SPECS:
        config = load_config(results_dir)
        history = load_history(results_dir)
        final_states_path = os.path.join(results_dir, "ns_final_states.npz")
        bundles.append({
            "name": name,
            "dir": results_dir,
            "config": config,
            "history": history,
            "final_states_path": final_states_path,
        })
    return bundles


def plot_residual(bundles, output_path):
    fig, axes = plt.subplots(
        1, len(bundles), figsize=(5.3 * len(bundles), 5.2),
        sharey=True, squeeze=False)
    legend_handles = {}

    for col_index, bundle in enumerate(bundles):
        ax = axes[0, col_index]
        history = bundle["history"]
        for method_key in ALGORITHM_ORDER:
            values = history.get(method_key, [])
            if method_key == "picard":
                values = [item for item in values if item["iter"] <= 100]
            if not values:
                continue
            label = values[0]["algorithm"]
            x_values = [item["iter"] for item in values]
            y_values = [item["rel_res"] for item in values]
            plot_style = dict(STYLE[label])
            if label in ("AA_pure", "FAA cs=1/sqrt(2)"):
                plot_style["markevery"] = marker_indices(x_values)
            line, = ax.semilogy(
                x_values,
                y_values,
                label=label,
                **plot_style,
            )
            legend_handles.setdefault(label, line)
            if label.startswith("GlobalAndersonSolver"):
                mark_last_km_step(ax, values, STYLE[label])

        ax.set_title(residual_title(bundle["config"]), fontsize=17, pad=12)
        ax.set_xlabel(r"Iteration $k$", fontsize=18)
        ax.set_xlim(-5.0, 105.0)
        ax.set_xticks(range(0, 101, 20))
        if col_index == 0:
            ax.set_ylabel(r"$\Vert F_k\Vert/\Vert F_0\Vert$", fontsize=18)
        ax.set_ylim(*Y_LIMITS)
        ax.tick_params(axis="both", labelsize=12)
        ax.yaxis.set_major_locator(FixedLocator(Y_MAJOR_TICKS))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.grid(True, which="both", linestyle="-", linewidth=0.6, alpha=0.35)

    handles = [legend_handles[label] for label in LEGEND_ORDER
               if label in legend_handles]
    labels = [display_label_for_m(label, ANDERSON_M) for label in LEGEND_ORDER
              if label in legend_handles]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            fontsize=15,
            handlelength=2.8,
            columnspacing=1.5,
            borderpad=0.45,
            bbox_to_anchor=(0.5, 0.985),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.88))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def sample_global_velocity(bundles):
    fields = []
    max_speed = 0.0
    for bundle in bundles:
        problem = build_problem(bundle["config"])
        with np.load(bundle["final_states_path"]) as final_states:
            if GLOBAL_FINAL_STATE_KEY not in final_states.files:
                raise ValueError(
                    f"{bundle['final_states_path']} is missing "
                    f"{GLOBAL_FINAL_STATE_KEY}.")
            velocity = final_states[GLOBAL_FINAL_STATE_KEY]
        grid, xx, yy, u, v = problem.sample_velocity(velocity)
        speed = np.sqrt(u * u + v * v)
        max_speed = max(max_speed, float(np.max(speed)))
        fields.append({
            "bundle": bundle,
            "grid": grid,
            "xx": xx,
            "yy": yy,
            "u": u,
            "v": v,
            "speed": speed,
        })
    return fields, max_speed


def plot_velocity(bundles, output_path):
    fields, max_speed = sample_global_velocity(bundles)
    global_label = display_label_for_m(
        "GlobalAndersonSolver", ANDERSON_M)
    fig, axes = plt.subplots(
        1, len(fields), figsize=(3.85 * len(fields), 4.35),
        sharex=True, sharey=True, squeeze=False)
    levels = np.linspace(0.0, max(max_speed, 1e-15), 31)
    contour = None

    for ax, field in zip(axes.flat, fields):
        contour = ax.contourf(
            field["xx"],
            field["yy"],
            field["speed"],
            levels=levels,
            cmap="viridis",
            extend="max",
        )
        ax.streamplot(
            field["grid"],
            field["grid"],
            field["u"],
            field["v"],
            color="white",
            density=1.1,
            linewidth=0.52,
            arrowsize=0.75,
        )
        title = result_title(field["bundle"]["config"])
        ax.set_title(
            f"{title}\n{global_label}",
            fontsize=10,
            pad=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*VELOCITY_AXIS_LIMITS)
        ax.set_ylim(*VELOCITY_AXIS_LIMITS)
        ax.set_xticks(VELOCITY_AXIS_TICKS)
        ax.set_yticks(VELOCITY_AXIS_TICKS)
        ax.set_xlabel("x", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
    axes[0, 0].set_ylabel("y", fontsize=11)

    colorbar_ax = fig.add_axes((0.92, 0.20, 0.012, 0.62))
    fig.colorbar(contour, cax=colorbar_ax, label="velocity magnitude")
    fig.subplots_adjust(
        left=0.055,
        right=0.90,
        bottom=0.13,
        top=0.80,
        wspace=0.18,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    bundles = load_result_bundle()
    residual_path = os.path.join(OUTPUT_DIR, RESIDUAL_OUTPUT)
    velocity_path = os.path.join(OUTPUT_DIR, VELOCITY_OUTPUT)
    plot_residual(bundles, residual_path)
    plot_velocity(bundles, velocity_path)
    print(f"Saved residual figure to {residual_path}")
    print(f"Saved velocity figure to {velocity_path}")


if __name__ == "__main__":
    main()
