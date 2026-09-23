"""
Run the Section 3.3 one-dimensional nonlinear Helmholtz experiment.

The NLH fixed-point map is complex-valued internally and is wrapped as a real
vector [Re(u), Im(u)] for the Anderson solvers. Write iteration histories and
summary CSV files to --output-dir; use plot_NLH.py to generate the figure.
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, replace
from time import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optim_algo import AndersonAcc, GlobalAndersonSolver


DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_M_VALUES = (1, 3, 5, 10, 20, 30)
HISTORY_FILENAME = "nlh_main_history.csv"
SUMMARY_FILENAME = "nlh_main_summary.csv"

ALGORITHMS = (
    "GlobalAndersonSolver",
    "GlobalAndersonSolver eta=1e-10",
    "AA_pure",
    "FAA cs=0.1",
    "FAA cs=0.2",
    "Picard",
)
REGULARIZED_GLOBAL_ETA = 1e-10
FAA_CONFIGS = (
    ("FAA cs=0.1", 0.1, 1e8),
    ("FAA cs=0.2", 0.2, 1e8),
)

HISTORY_FIELDS = (
    "algorithm",
    "m",
    "iter",
    "rel_res",
    "residual",
    "time",
    "step_type",
    "effective_m",
)
SUMMARY_FIELDS = (
    "algorithm",
    "m",
    "iterations",
    "time",
    "final_rel_res",
    "final_residual",
    "aa_steps",
    "aa_rate_percent",
)


@dataclass(frozen=True)
class NLHProblem:
    length: float = 10.0
    k0: float = 8.0
    epsilon: float = 0.2
    N: int = 2001

    def __post_init__(self):
        if self.N < 3:
            raise ValueError("N must be at least 3 for the second-order boundary stencils.")
        if self.length <= 0:
            raise ValueError("length must be positive.")
        if self.k0 <= 0:
            raise ValueError("k0 must be positive.")
        if self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative.")

    @property
    def h(self):
        return self.length / (self.N - 1)

    @property
    def x(self):
        return np.linspace(0.0, self.length, self.N, dtype=np.float64)

    @property
    def recommended_m(self):
        return 20

    def initial_guess_complex(self):
        return np.exp(1j * self.k0 * self.x).astype(np.complex128, copy=False)

    def rhs(self):
        b = np.zeros(self.N, dtype=np.complex128)
        b[0] = 2j * self.k0
        return b

    def matrix(self, u):
        u = self._as_complex_vector(u)
        h = self.h
        inv_h2 = 1.0 / (h * h)

        main = np.zeros(self.N, dtype=np.complex128)
        lower1 = np.zeros(self.N - 1, dtype=np.complex128)
        upper1 = np.zeros(self.N - 1, dtype=np.complex128)
        lower2 = np.zeros(self.N - 2, dtype=np.complex128)
        upper2 = np.zeros(self.N - 2, dtype=np.complex128)

        main[1:-1] = (
            -2.0 * inv_h2
            + self.k0 * self.k0 * (1.0 + self.epsilon * np.abs(u[1:-1]) ** 2)
        )
        lower1[:-1] = inv_h2
        upper1[1:] = inv_h2

        main[0] = -3.0 / (2.0 * h) + 1j * self.k0
        upper1[0] = 2.0 / h
        upper2[0] = -1.0 / (2.0 * h)

        lower2[-1] = 1.0 / (2.0 * h)
        lower1[-1] = -2.0 / h
        main[-1] = 3.0 / (2.0 * h) - 1j * self.k0

        return sp.diags(
            diagonals=(lower2, lower1, main, upper1, upper2),
            offsets=(-2, -1, 0, 1, 2),
            shape=(self.N, self.N),
            format="csc",
        )

    def fixed_point_map_complex(self, u):
        A = self.matrix(u)
        return spla.spsolve(A, self.rhs()).astype(np.complex128, copy=False)

    def residual_complex(self, u):
        u = self._as_complex_vector(u)
        return self.fixed_point_map_complex(u) - u

    def residual_norm_complex(self, u):
        return np.linalg.norm(self.residual_complex(u))

    def pack_real(self, u):
        u = self._as_complex_vector(u)
        return np.concatenate((u.real, u.imag)).astype(np.float64, copy=False)

    def unpack_real(self, y):
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (2 * self.N,):
            raise ValueError(f"Expected a real vector of length {2 * self.N}, got {y.shape}.")
        return y[:self.N] + 1j * y[self.N:]

    def initial_guess_real_torch(self):
        return torch.as_tensor(self.pack_real(self.initial_guess_complex()),
                               dtype=torch.float64)

    def fixed_point_map_real_torch(self, y):
        u = self.unpack_real(y)
        Gy = self.pack_real(self.fixed_point_map_complex(u))
        return torch.as_tensor(Gy, dtype=y.dtype, device=y.device)

    def _as_complex_vector(self, u):
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.N,):
            raise ValueError(f"Expected a complex vector of length {self.N}, got {u.shape}.")
        return u


@dataclass
class SolverConfig:
    max_iters: int
    func_eval_max: int
    tol: float
    lambda_: float
    eta: float
    show: bool
    pure_aa_restart: bool = False
    gamma_k: object = "paper_window"


def parse_number_list(value, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative.")
    return parsed


def run_global_anderson_solver(nlh_map, x0, anderson_m, config):
    if config.gamma_k is None:
        gamma_k = "paper_window"
    else:
        gamma_k = config.gamma_k

    solver = GlobalAndersonSolver(
        nlh_map,
        x0.clone(),
        m=anderson_m,
        lambda_=config.lambda_,
        mainLoopMaxItrs=config.max_iters,
        funcEvalMax=config.func_eval_max,
        gamma_k=gamma_k,
        residualTol=config.tol,
        eta=config.eta,
        show=config.show,
        print_final=False,
    )
    return solver.run()


def config_with_eta(config, eta):
    return replace(config, eta=eta)


def run_pure_aa(nlh_map, x0, anderson_m, config):
    x, record = AndersonAcc(
        nlh_map,
        x0.clone(),
        anderson_m,
        config.lambda_,
        1.0,
        config.max_iters,
        config.func_eval_max,
        0,
        0,
        0,
        0,
        0,
        gradTol=config.tol,
        show=config.show,
        arg="pure",
        restart=config.pure_aa_restart,
        regularization_eta=config.eta,
        print_final=False,
    )
    return x, record


def run_faa(nlh_map, x0, anderson_m, config, cs, kappa_bar):
    x, record = AndersonAcc(
        nlh_map,
        x0.clone(),
        anderson_m,
        config.lambda_,
        1.0,
        config.max_iters,
        config.func_eval_max,
        0,
        0,
        0,
        0,
        0,
        gradTol=config.tol,
        show=config.show,
        arg="pure",
        restart=False,
        regularization_eta=0.0,
        print_final=False,
        coefficient_method="faa",
        cs=cs,
        kappa_bar=kappa_bar,
    )
    return x, record


def run_picard(nlh_map, x0, config):
    x = x0.clone()
    tmk = 0.0
    iters = 0

    Sx = nlh_map(x).detach()
    Fx = Sx - x
    residual_norm = Fx.norm()
    residual0_norm = torch.clamp(
        residual_norm.detach().clone(), min=torch.finfo(x.dtype).tiny)
    record = torch.tensor(
        [residual_norm / residual0_norm, residual_norm, 0, tmk, 1],
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, -1)

    while residual_norm / residual0_norm >= config.tol \
            and iters < config.max_iters:
        t0 = time()
        x = Sx.clone()
        Sx = nlh_map(x).detach()
        Fx = Sx - x
        residual_norm = Fx.norm()
        iters += 1
        tmk += time() - t0
        row = torch.tensor(
            [residual_norm / residual0_norm, residual_norm, 0, tmk, 1],
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, -1)
        record = torch.cat((record, row), axis=0)

    return x, record


def step_type_for_row(algorithm, row_values, iter_index):
    if iter_index == 0:
        return "initial"

    direction_value = int(round(row_values[-1]))
    if algorithm.startswith("GlobalAndersonSolver"):
        return "AA" if direction_value == 0 else "KM"
    if algorithm == "Picard":
        return "Picard"
    return "AA" if direction_value == 0 else "Picard"


def effective_m_for_row(algorithm, row_values, iter_index, anderson_m):
    if algorithm.startswith("GlobalAndersonSolver"):
        return int(round(row_values[7]))
    if algorithm.startswith("FAA"):
        return int(round(row_values[4]))
    if algorithm == "AA_pure":
        return min(iter_index, anderson_m)
    return 0


def record_to_history_rows(algorithm, anderson_m, record):
    values = record.detach().cpu().numpy()
    rows = []
    for iter_index, row in enumerate(values):
        rows.append({
            "algorithm": algorithm,
            "m": anderson_m,
            "iter": iter_index,
            "rel_res": float(row[0]),
            "residual": float(row[1]),
            "time": float(row[3]),
            "step_type": step_type_for_row(algorithm, row, iter_index),
            "effective_m": effective_m_for_row(
                algorithm, row, iter_index, anderson_m),
        })
    return rows


def summarize_history_rows(algorithm, anderson_m, rows):
    final = rows[-1]
    iterations = int(final["iter"])
    aa_steps = sum(
        1 for row in rows if row["iter"] > 0 and row["step_type"] == "AA")
    aa_rate_percent = 100.0 * aa_steps / max(iterations, 1)
    return {
        "algorithm": algorithm,
        "m": anderson_m,
        "iterations": iterations,
        "time": final["time"],
        "final_rel_res": final["rel_res"],
        "final_residual": final["residual"],
        "aa_steps": aa_steps,
        "aa_rate_percent": aa_rate_percent,
    }


def print_summary_header(anderson_m):
    print("-" * 116)
    print(f"m = {anderson_m}")
    print(
        f"  {'algorithm':<36} {'iterations':>10} {'time(s)':>12} "
        f"{'rel_res':>12} {'residual':>12} {'AA_steps':>9} {'AA_rate%':>10}"
    )
    print("-" * 116)


def print_summary_row(summary):
    print(
        f"  {summary['algorithm']:<36} {summary['iterations']:>10d} "
        f"{summary['time']:>12.2f} {summary['final_rel_res']:>12.4e} "
        f"{summary['final_residual']:>12.4e} {summary['aa_steps']:>9d} "
        f"{summary['aa_rate_percent']:>10.2f}"
    )


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_and_record(algorithm, run_callable, anderson_m, history_rows,
                   summary_rows):
    _, record = run_callable()
    rows = record_to_history_rows(algorithm, anderson_m, record)
    summary = summarize_history_rows(algorithm, anderson_m, rows)
    history_rows.extend(rows)
    summary_rows.append(summary)
    print_summary_row(summary)
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal NLH fixed-point experiment for Anderson acceleration.")
    parser.add_argument("--N", type=int, default=2001,
                        help="Grid points. The formal main setting uses N=2001.")
    parser.add_argument("--m-values", default="1,3,5,10,20,30",
                        help="Comma-separated Anderson memory values.")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--func-eval-max", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--lambda", "--lambda_", dest="lambda_", type=float,
                        default=0.5)
    parser.add_argument(
        "--eta", type=nonnegative_float, default=0.0,
        help="Regularization parameter for the base GlobalAndersonSolver run. "
             "AA_pure is kept unregularized; the additional regularized "
             "GlobalAndersonSolver uses eta=1e-10.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    args.m_values = parse_number_list(args.m_values, int)
    if not args.m_values:
        raise ValueError("--m-values must contain at least one integer.")
    for anderson_m in args.m_values:
        if anderson_m <= 0:
            raise ValueError("All m values must be positive integers.")
    if args.N < 3:
        raise ValueError("N must be at least 3.")
    if args.func_eval_max is None:
        args.func_eval_max = 2 * args.max_iters + 2
    return args


def main():
    args = parse_args()
    problem = NLHProblem(length=10.0, k0=8.0, epsilon=0.2, N=args.N)
    config = SolverConfig(
        max_iters=args.max_iters,
        func_eval_max=args.func_eval_max,
        tol=args.tol,
        lambda_=args.lambda_,
        eta=args.eta,
        show=args.show,
    )
    nlh_map = problem.fixed_point_map_real_torch
    x0 = problem.initial_guess_real_torch()

    print("Nonlinear Helmholtz fixed-point formal experiment")
    print(f"  setting=main, interval=[0, {problem.length:g}]")
    print(f"  k0={problem.k0:g}, epsilon={problem.epsilon:g}")
    print(f"  N={problem.N}, h={problem.h:.6g}, real dimension={2 * problem.N}")
    print("  initial guess=exp(1j*k0*x)")
    print("  gamma_k=1-max{1/(k+2), W_k/(2W_0)}")
    print(f"  base eta={config.eta:.4e}, regularized Global eta="
          f"{REGULARIZED_GLOBAL_ETA:.4e}, lambda={config.lambda_:.4e}")
    print(f"  tol={config.tol:.1e}, max_iters={config.max_iters}")
    print(f"  m values={args.m_values}")
    print(f"  algorithms={', '.join(ALGORITHMS)}")
    print("  FAA kappa_bar=1.0000e+08, FAA regularization_eta=0")
    print("  stopping criterion=fixed-point relative residual")

    history_rows = []
    summary_rows = []
    picard_record = None

    for anderson_m in args.m_values:
        print_summary_header(anderson_m)
        run_and_record(
            "GlobalAndersonSolver",
            lambda m=anderson_m: run_global_anderson_solver(
                nlh_map, x0, m, config),
            anderson_m,
            history_rows,
            summary_rows,
        )
        run_and_record(
            "GlobalAndersonSolver eta=1e-10",
            lambda m=anderson_m: run_global_anderson_solver(
                nlh_map, x0, m,
                config_with_eta(config, REGULARIZED_GLOBAL_ETA)),
            anderson_m,
            history_rows,
            summary_rows,
        )
        run_and_record(
            "AA_pure",
            lambda m=anderson_m: run_pure_aa(
                nlh_map, x0, m, config_with_eta(config, 0.0)),
            anderson_m,
            history_rows,
            summary_rows,
        )
        for faa_label, faa_cs, faa_kappa_bar in FAA_CONFIGS:
            run_and_record(
                faa_label,
                lambda m=anderson_m, cs=faa_cs, kappa=faa_kappa_bar: run_faa(
                    nlh_map, x0, m, config, cs, kappa),
                anderson_m,
                history_rows,
                summary_rows,
            )
        if picard_record is None:
            _, picard_record = run_picard(nlh_map, x0, config)
        picard_rows = record_to_history_rows("Picard", anderson_m, picard_record)
        picard_summary = summarize_history_rows(
            "Picard", anderson_m, picard_rows)
        history_rows.extend(picard_rows)
        summary_rows.append(picard_summary)
        print_summary_row(picard_summary)

    history_path = os.path.join(args.output_dir, HISTORY_FILENAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILENAME)
    write_csv(history_path, history_rows, HISTORY_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"Saved history CSV to {history_path}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
