"""
Run a Section 3.4 Navier-Stokes lid-driven cavity experiment.

The model uses grad-div stabilized Taylor-Hood P2/P1 finite elements and the
shared Anderson.py / optim_algo.py solvers. Write CSV histories and summaries,
NPZ final states, and a JSON configuration to --output-dir. See README.md for
the three experiment commands and use plot_NS_report_basepoints.py to plot
their saved results.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from time import perf_counter
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
    asm,
    condense,
)
from skfem.helpers import ddot, div, dot, grad

from optim_algo import AndersonAcc, GlobalAndersonSolver


DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
HISTORY_FILENAME = "ns_main_history.csv"
SUMMARY_FILENAME = "ns_main_summary.csv"
FINAL_STATES_FILENAME = "ns_final_states.npz"
CONFIG_FILENAME = "ns_run_config.json"
PAPER_REPORTED_DOFS = 190_643
REGULARIZED_GLOBAL_ETA = 1e-10
FAA_KAPPA_BAR = 1e8
GLOBAL_GAMMA_STRATEGY = "paper_window"
GLOBAL_GAMMA_FORMULA = "1-max{1/(k+2), W_k/(2W_0)}"

HISTORY_FIELDS = (
    "method_key",
    "algorithm",
    "m",
    "iter",
    "time",
    "rel_res",
    "residual",
    "step_type",
    "AA_steps",
    "AA_percent",
    "eta",
    "cs",
    "effective_m",
    "bu_norm",
)
SUMMARY_FIELDS = (
    "method_key",
    "algorithm",
    "m",
    "iterations",
    "time",
    "final_rel_res",
    "final_residual",
    "AA_steps",
    "AA_percent",
    "eta",
    "cs",
    "effective_m",
    "bu_norm",
)


@BilinearForm
def viscous_grad_div(u, v, w):
    return (
        w.nu * ddot(grad(u), grad(v))
        + w.grad_div * div(u) * div(v)
    )


@BilinearForm
def oseen_convection(u, v, w):
    advect_trial = np.einsum("j...,ij...->i...", w.u_old, grad(u))
    advect_test = np.einsum("j...,ij...->i...", w.u_old, grad(v))
    return 0.5 * (dot(advect_trial, v) - dot(advect_test, u))


@BilinearForm
def negative_divergence(u, q, w):
    return -q * div(u)


def build_paper_mesh(base_points, edge_width, refine_edges):
    coordinates = np.linspace(0.0, 1.0, base_points)
    mesh = MeshTri.init_tensor(coordinates, coordinates)
    marked_count = 0
    if refine_edges:
        centers = mesh.p[:, mesh.t].mean(axis=1)
        distance_to_boundary = np.min(
            np.vstack((centers, 1.0 - centers)), axis=0)
        marked = np.flatnonzero(distance_to_boundary < edge_width)
        marked_count = len(marked)
        mesh = mesh.refined(marked)
    return mesh, marked_count


@dataclass
class TaylorHoodCavityProblem:
    base_points: int = 10
    reynolds: float = 10000.0
    grad_div: float = 1.0
    edge_width: float = 0.1
    refine_edges: bool = True
    quadrature_order: int = 6

    def __post_init__(self):
        if self.base_points < 3:
            raise ValueError("base_points must be at least 3.")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive.")
        if self.grad_div < 0.0:
            raise ValueError("grad_div must be nonnegative.")
        if not 0.0 < self.edge_width < 0.5:
            raise ValueError("edge_width must lie between 0 and 0.5.")

        start = perf_counter()
        self.mesh, self.marked_elements = build_paper_mesh(
            self.base_points, self.edge_width, self.refine_edges)
        self.velocity_basis = Basis(
            self.mesh,
            ElementVector(ElementTriP2()),
            intorder=self.quadrature_order,
        )
        self.pressure_basis = Basis(
            self.mesh,
            ElementTriP1(),
            intorder=self.quadrature_order,
        )
        self.velocity_dofs = self.velocity_basis.N
        self.pressure_dofs = self.pressure_basis.N
        self.mixed_dofs = self.velocity_dofs + self.pressure_dofs

        self._static_velocity = asm(
            viscous_grad_div,
            self.velocity_basis,
            nu=1.0 / self.reynolds,
            grad_div=self.grad_div,
        ).tocsr()
        self._divergence = asm(
            negative_divergence,
            self.velocity_basis,
            self.pressure_basis,
        ).tocsr()
        self._pressure_zero = sp.csr_matrix(
            (self.pressure_dofs, self.pressure_dofs))
        self._boundary_values, self._dirichlet_dofs = (
            self._build_dirichlet_data())
        self.last_pressure = np.zeros(self.pressure_dofs)
        self.setup_time = perf_counter() - start

    @property
    def dimension(self):
        return self.velocity_dofs

    def initial_guess(self):
        return np.zeros(self.velocity_dofs, dtype=np.float64)

    def _build_dirichlet_data(self):
        boundary = self.velocity_basis.get_dofs()
        velocity_boundary_dofs = boundary.all()
        values = np.zeros(self.mixed_dofs, dtype=np.float64)

        x_component_dofs = np.concatenate((
            boundary.nodal["u^1"],
            boundary.facet["u^1"],
        ))
        locations = self.velocity_basis.doflocs[:, x_component_dofs]
        on_open_lid = (
            np.isclose(locations[1], 1.0)
            & (locations[0] > 1e-12)
            & (locations[0] < 1.0 - 1e-12)
        )
        values[x_component_dofs[on_open_lid]] = 1.0

        pressure_pin = self.velocity_dofs
        dirichlet_dofs = np.concatenate((
            velocity_boundary_dofs,
            np.array([pressure_pin], dtype=np.int64),
        ))
        return values, np.unique(dirichlet_dofs)

    def fixed_point_map(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.velocity_dofs,):
            raise ValueError(
                f"Expected velocity vector of length {self.velocity_dofs}, "
                f"got {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Picard input contains non-finite values.")

        convection = asm(
            oseen_convection,
            self.velocity_basis,
            u_old=self.velocity_basis.interpolate(x),
        ).tocsr()
        velocity_block = self._static_velocity + convection
        saddle_matrix = sp.bmat(
            [
                [velocity_block, self._divergence.T],
                [self._divergence, self._pressure_zero],
            ],
            format="csc",
        )
        rhs = np.zeros(self.mixed_dofs, dtype=np.float64)
        reduced_matrix, reduced_rhs, solution, free_dofs = condense(
            saddle_matrix,
            rhs,
            x=self._boundary_values.copy(),
            D=self._dirichlet_dofs,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=spla.MatrixRankWarning)
            reduced_solution = spla.spsolve(reduced_matrix, reduced_rhs)
        if not np.all(np.isfinite(reduced_solution)):
            raise FloatingPointError(
                "The Taylor-Hood Oseen solve produced non-finite values.")

        solution[free_dofs] = reduced_solution
        velocity = solution[:self.velocity_dofs].copy()
        self.last_pressure = solution[self.velocity_dofs:].copy()
        return velocity

    def fixed_point_map_torch(self, x):
        velocity = x.detach().cpu().numpy()
        mapped = self.fixed_point_map(velocity)
        return torch.as_tensor(mapped, dtype=x.dtype, device=x.device)

    def discrete_divergence_norm(self, velocity):
        return float(np.linalg.norm(self._divergence @ velocity))

    def dirichlet_boundary_error(self, velocity):
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (self.velocity_dofs,):
            raise ValueError(
                f"Expected velocity vector of length {self.velocity_dofs}, "
                f"got {velocity.shape}.")
        boundary_dofs = self._dirichlet_dofs[
            self._dirichlet_dofs < self.velocity_dofs]
        target = self._boundary_values[:self.velocity_dofs]
        return float(np.max(np.abs(
            velocity[boundary_dofs] - target[boundary_dofs])))

    def sample_velocity(self, velocity, plot_points=101):
        grid = np.linspace(0.0, 1.0, plot_points)
        xx, yy = np.meshgrid(grid, grid)
        points = np.vstack((xx.ravel(), yy.ravel()))
        interior_points = points.copy()
        eps = 16.0 * np.finfo(np.float64).eps
        interior_points[:] = np.clip(interior_points, eps, 1.0 - eps)

        split_fields = self.velocity_basis.split(velocity)
        sampled = []
        for coefficients, scalar_basis in split_fields:
            values = scalar_basis.probes(interior_points) @ coefficients
            sampled.append(np.asarray(values).reshape(xx.shape))
        u, v = sampled

        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = 1.0
        u[-1, 0] = 0.0
        u[-1, -1] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        return grid, xx, yy, u, v


@dataclass(frozen=True)
class SolverConfig:
    max_iters: int
    func_eval_max: int
    tol: float
    lambda_: float
    show: bool


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    kind: str
    eta: float = 0.0
    cs: float = math.nan
    kappa_bar: float = FAA_KAPPA_BAR


ALGORITHM_SPECS = (
    AlgorithmSpec("picard", "Picard", "picard"),
    AlgorithmSpec("aa_pure", "AA_pure", "aa"),
    AlgorithmSpec("global_eta0", "GlobalAndersonSolver", "global", eta=0.0),
    AlgorithmSpec(
        "global_eta1e10",
        "GlobalAndersonSolver eta=1e-10",
        "global",
        eta=REGULARIZED_GLOBAL_ETA,
    ),
    AlgorithmSpec("faa02", "FAA cs=0.2", "faa", cs=0.2),
    AlgorithmSpec(
        "faa0707",
        "FAA cs=1/sqrt(2)",
        "faa",
        cs=1.0 / math.sqrt(2.0),
    ),
)


class IterationRecorder:
    def __init__(self, problem):
        self.problem = problem
        self.start = perf_counter()
        self.times = []
        self.bu_norms = []

    def __call__(self, iteration, x):
        del iteration
        self.times.append(perf_counter() - self.start)
        if torch.is_tensor(x):
            velocity = x.detach().cpu().numpy()
        else:
            velocity = np.asarray(x, dtype=np.float64)
        self.bu_norms.append(self.problem.discrete_divergence_norm(velocity))


def build_problem_from_args(args):
    return TaylorHoodCavityProblem(
        base_points=args.base_points,
        reynolds=args.re,
        grad_div=args.grad_div,
        edge_width=args.edge_width,
        refine_edges=not args.no_edge_refine,
        quadrature_order=args.quadrature_order,
    )


def build_common_initial_point(problem):
    start = perf_counter()
    initial = problem.initial_guess()
    common_initial = problem.fixed_point_map(initial)
    elapsed = perf_counter() - start
    boundary_error = problem.dirichlet_boundary_error(common_initial)
    if boundary_error > 1e-12:
        raise RuntimeError(
            "The common Picard initial point violates the velocity "
            f"Dirichlet boundary condition: max error={boundary_error:.4e}.")
    return common_initial, elapsed


def run_global(problem, spec, anderson_m, args, config, initial_x):
    recorder = IterationRecorder(problem)
    x0 = torch.as_tensor(initial_x.copy(), dtype=torch.float64)

    def record_global_iterate(iteration, x):
        velocity = x.detach().cpu().numpy()
        boundary_error = problem.dirichlet_boundary_error(velocity)
        if boundary_error > 1e-12:
            raise RuntimeError(
                "Global Anderson violated the velocity Dirichlet boundary "
                f"condition at iteration {iteration}: "
                f"max error={boundary_error:.4e}.")
        recorder(iteration, x)

    solver = GlobalAndersonSolver(
        problem.fixed_point_map_torch,
        x0,
        m=anderson_m,
        lambda_=config.lambda_,
        mainLoopMaxItrs=config.max_iters,
        funcEvalMax=config.func_eval_max,
        gamma_k=GLOBAL_GAMMA_STRATEGY,
        residualTol=config.tol,
        eta=spec.eta,
        show=config.show,
        print_final=False,
        iterate_callback=record_global_iterate,
    )
    final_x, record = solver.run()
    rows = record_to_history_rows(spec, anderson_m, record, recorder)
    return final_x.detach().cpu().numpy(), rows


def run_aa(problem, spec, anderson_m, args, config, initial_x):
    recorder = IterationRecorder(problem)
    x0 = torch.as_tensor(initial_x.copy(), dtype=torch.float64)
    coefficient_kwargs = {}
    coefficient_method = "pure"
    regularization_eta = 0.0
    if spec.kind == "faa":
        coefficient_method = "faa"
        coefficient_kwargs = {
            "cs": spec.cs,
            "kappa_bar": spec.kappa_bar,
        }

    final_x, record = AndersonAcc(
        problem.fixed_point_map_torch,
        x0,
        anderson_m,
        1.0,
        1.0,
        config.max_iters,
        config.func_eval_max,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        gradTol=config.tol,
        show=config.show,
        arg="pure",
        restart=False,
        regularization_eta=regularization_eta,
        print_final=False,
        iterate_callback=recorder,
        coefficient_method=coefficient_method,
        **coefficient_kwargs,
    )
    rows = record_to_history_rows(spec, anderson_m, record, recorder)
    return final_x.detach().cpu().numpy(), rows


def run_picard(problem, spec, anderson_m, args, config, initial_x):
    start = perf_counter()
    x = initial_x.copy()
    rows = []
    residual0 = None
    aa_steps = 0
    for iteration in range(config.max_iters + 1):
        mapped = problem.fixed_point_map(x)
        residual = float(np.linalg.norm(mapped - x))
        if residual0 is None:
            residual0 = max(residual, np.finfo(np.float64).tiny)
        rel_res = residual / residual0
        elapsed = perf_counter() - start
        rows.append({
            "method_key": spec.key,
            "algorithm": spec.label,
            "m": anderson_m,
            "iter": iteration,
            "time": elapsed,
            "rel_res": rel_res,
            "residual": residual,
            "step_type": "Picard",
            "AA_steps": aa_steps,
            "AA_percent": 0.0,
            "eta": spec.eta,
            "cs": "",
            "effective_m": 0,
            "bu_norm": problem.discrete_divergence_norm(x),
        })
        if rel_res < config.tol or iteration >= config.max_iters:
            return x.copy(), rows
        x = mapped
    return x.copy(), rows


def run_algorithm(problem, spec, anderson_m, args, config, initial_x):
    if spec.kind == "picard":
        return run_picard(
            problem, spec, anderson_m, args, config, initial_x)
    if spec.kind == "global":
        return run_global(
            problem, spec, anderson_m, args, config, initial_x)
    if spec.kind in ("aa", "faa"):
        return run_aa(
            problem, spec, anderson_m, args, config, initial_x)
    raise ValueError(f"Unknown algorithm kind: {spec.kind}")


def step_type_for_row(spec, row, iter_index):
    if iter_index == 0:
        return "initial"
    if spec.kind == "global":
        return "KM" if int(round(row[9])) == 1 else "AA"
    if spec.kind in ("aa", "faa"):
        return "Picard" if int(round(row[-1])) == 1 else "AA"
    return "Picard"


def effective_m_for_row(spec, row, iter_index, anderson_m):
    if spec.kind == "global":
        return int(round(row[7]))
    if spec.kind == "faa":
        return int(round(row[4]))
    if spec.kind == "aa":
        return min(iter_index, anderson_m)
    return 0


def record_to_history_rows(spec, anderson_m, record, recorder):
    values = record.detach().cpu().numpy()
    rows = []
    aa_steps = 0
    for iter_index, row in enumerate(values):
        step_type = step_type_for_row(spec, row, iter_index)
        if iter_index > 0 and step_type == "AA":
            aa_steps += 1
        aa_percent = 100.0 * aa_steps / max(iter_index, 1)
        elapsed = recorder.times[iter_index] if iter_index < len(
            recorder.times) else float(row[3])
        bu_norm = recorder.bu_norms[iter_index] if iter_index < len(
            recorder.bu_norms) else math.nan
        rows.append({
            "method_key": spec.key,
            "algorithm": spec.label,
            "m": anderson_m,
            "iter": iter_index,
            "time": float(elapsed),
            "rel_res": float(row[0]),
            "residual": float(row[1]),
            "step_type": step_type,
            "AA_steps": aa_steps,
            "AA_percent": aa_percent,
            "eta": spec.eta,
            "cs": "" if math.isnan(spec.cs) else spec.cs,
            "effective_m": effective_m_for_row(
                spec, row, iter_index, anderson_m),
            "bu_norm": float(bu_norm),
        })
    return rows


def summarize_history_rows(rows):
    final = rows[-1]
    return {
        "method_key": final["method_key"],
        "algorithm": final["algorithm"],
        "m": final["m"],
        "iterations": final["iter"],
        "time": final["time"],
        "final_rel_res": final["rel_res"],
        "final_residual": final["residual"],
        "AA_steps": final["AA_steps"],
        "AA_percent": final["AA_percent"],
        "eta": final["eta"],
        "cs": final["cs"],
        "effective_m": final["effective_m"],
        "bu_norm": final["bu_norm"],
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(path, args, problem, common_initialization_time):
    algorithms = []
    for spec in ALGORITHM_SPECS:
        item = asdict(spec)
        if math.isnan(item["cs"]):
            item["cs"] = None
        algorithms.append(item)

    config = {
        "schema_version": 1,
        "history_filename": HISTORY_FILENAME,
        "summary_filename": SUMMARY_FILENAME,
        "final_states_filename": FINAL_STATES_FILENAME,
        "m_values": [int(value) for value in args.m_values],
        "max_iters": args.max_iters,
        "func_eval_max": args.func_eval_max,
        "tol": args.tol,
        "global_lambda": args.global_lambda,
        "global_gamma_strategy": GLOBAL_GAMMA_STRATEGY,
        "global_gamma_formula": GLOBAL_GAMMA_FORMULA,
        "common_initial_point": "G(0)",
        "common_picard_initialization": True,
        "common_initialization_time": float(common_initialization_time),
        "algorithm_times_include_common_initialization": False,
        "algorithm_iterations_include_common_initialization": False,
        "base_points": int(args.base_points),
        "re": float(args.re),
        "grad_div": float(args.grad_div),
        "edge_width": float(args.edge_width),
        "refine_edges": not args.no_edge_refine,
        "quadrature_order": int(args.quadrature_order),
        "velocity_dofs": int(problem.velocity_dofs),
        "pressure_dofs": int(problem.pressure_dofs),
        "mixed_dofs": int(problem.mixed_dofs),
        "mesh_vertices": int(problem.mesh.nvertices),
        "mesh_triangles": int(problem.mesh.nelements),
        "marked_elements": int(problem.marked_elements),
        "algorithms": algorithms,
    }
    with open(path, "w") as json_file:
        json.dump(config, json_file, indent=2, allow_nan=False)


def print_problem_report(problem):
    print("Taylor-Hood cavity problem")
    print(f"  Re={problem.reynolds:g}, grad-div={problem.grad_div:g}")
    print(f"  base points/direction={problem.base_points}")
    print(f"  boundary elements marked={problem.marked_elements:,}")
    print(f"  vertices={problem.mesh.nvertices:,}")
    print(f"  triangles={problem.mesh.nelements:,}")
    print(f"  velocity P2 DOFs={problem.velocity_dofs:,}")
    print(f"  pressure P1 DOFs={problem.pressure_dofs:,}")
    print(f"  total mixed DOFs={problem.mixed_dofs:,}")
    print(f"  setup time={problem.setup_time:.2f}s")


def format_eta(value):
    if float(value) == 0.0:
        return "0"
    return f"{float(value):.0e}"


def console_algorithm_name(summary):
    method_key = summary["method_key"]
    eta_text = format_eta(summary["eta"])
    if method_key == "picard":
        return "Picard"
    if method_key == "aa_pure":
        return f"AA_pure, eta={eta_text}"
    if method_key.startswith("global"):
        return f"Global AA, eta={eta_text}"
    if method_key == "faa02":
        return "FAA cs=0.2"
    if method_key == "faa0707":
        return "FAA cs=1/sqrt(2)"
    return summary["algorithm"]


def print_summary_header(anderson_m):
    print("-" * 94)
    print(f"m = {anderson_m}")
    print(
        f"  {'algorithm':<24} {'iter':>5} "
        f"{'time':>7} {'rel_res':>10} {'res':>10} "
        f"{'AA':>5} {'AA%':>6} {'||Bu||':>10}"
    )
    print("-" * 94)


def print_summary_row(summary):
    algorithm = console_algorithm_name(summary)
    print(
        f"  {algorithm:<24} {summary['iterations']:>5d} "
        f"{summary['time']:>7.2f} "
        f"{summary['final_rel_res']:>10.3e} "
        f"{summary['final_residual']:>10.3e} "
        f"{summary['AA_steps']:>5d} {summary['AA_percent']:>6.1f} "
        f"{summary['bu_norm']:>10.3e}"
    )


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return parsed


def parse_m_values(value):
    values = tuple(
        int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "--m-values must contain at least one integer.")
    for item in values:
        if item <= 0:
            raise argparse.ArgumentTypeError(
                "All m values must be positive integers.")
    return values


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative.")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal P2/P1 Taylor-Hood NS cavity experiment.")
    parser.add_argument("--base-points", type=positive_int, default=10)
    parser.add_argument("--re", type=float, default=5000.0)
    parser.add_argument("--grad-div", type=nonnegative_float, default=1.0)
    parser.add_argument("--edge-width", type=float, default=0.1)
    parser.add_argument("--no-edge-refine", action="store_true")
    parser.add_argument("--quadrature-order", type=positive_int, default=6)
    parser.add_argument("--m-values", type=parse_m_values, default=(10, 20, 30, 50))
    parser.add_argument("--max-iters", type=positive_int, default=100)
    parser.add_argument("--func-eval-max", type=positive_int, default=None)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--global-lambda", type=float, default=0.5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.re <= 0.0:
        raise ValueError("--re must be positive.")
    if not 0.0 < args.global_lambda <= 1.0:
        raise ValueError("--global-lambda must be in (0, 1].")
    if args.tol <= 0.0:
        raise ValueError("--tol must be positive.")
    if args.func_eval_max is None:
        args.func_eval_max = 3 * args.max_iters + 3
    return args


def rows_with_m(rows, anderson_m):
    return [{**row, "m": anderson_m} for row in rows]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    problem = build_problem_from_args(args)
    print_problem_report(problem)
    common_initial_x, common_initialization_time = (
        build_common_initial_point(problem))
    print(
        f"\nSolving with m values={args.m_values}, tol={args.tol:.1e}, "
        f"max_iters={args.max_iters}")
    print(
        "Common initial point=G(0), "
        f"initialization time={common_initialization_time:.2f}s "
        "(excluded from per-algorithm times)")
    print(f"Global lambda={args.global_lambda:.4g}")
    print(f"Global gamma_k={GLOBAL_GAMMA_FORMULA}")
    print(f"FAA kappa_bar={FAA_KAPPA_BAR:.4e}, FAA regularization_eta=0")

    config = SolverConfig(
        max_iters=args.max_iters,
        func_eval_max=args.func_eval_max,
        tol=args.tol,
        lambda_=args.global_lambda,
        show=args.show,
    )

    history_rows = []
    summary_rows = []
    final_states = {}
    picard_final = None
    picard_rows = None

    for anderson_m in args.m_values:
        print_summary_header(anderson_m)
        for spec in ALGORITHM_SPECS:
            if spec.kind == "picard" and picard_rows is not None:
                final_x = picard_final
                rows = rows_with_m(picard_rows, anderson_m)
            else:
                final_x, rows = run_algorithm(
                    problem, spec, anderson_m, args, config,
                    common_initial_x)
                if spec.kind == "picard":
                    picard_final = final_x
                    picard_rows = rows
            summary = summarize_history_rows(rows)
            history_rows.extend(rows)
            summary_rows.append(summary)
            final_states[f"m{anderson_m}_{spec.key}"] = final_x
            print_summary_row(summary)

    history_path = os.path.join(args.output_dir, HISTORY_FILENAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILENAME)
    final_states_path = os.path.join(args.output_dir, FINAL_STATES_FILENAME)
    config_path = os.path.join(args.output_dir, CONFIG_FILENAME)
    write_csv(history_path, history_rows, HISTORY_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    np.savez(final_states_path, **final_states)
    write_run_config(
        config_path, args, problem, common_initialization_time)

    print("-" * 94)
    print(f"Saved history CSV to {history_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved final states to {final_states_path}")
    print(f"Saved run config to {config_path}")


if __name__ == "__main__":
    main()
