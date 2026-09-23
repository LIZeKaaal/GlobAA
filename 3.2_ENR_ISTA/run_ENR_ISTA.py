"""
Run the Section 3.2 elastic-net regression experiments with ISTA.

The two cases use the bundled Madelon training matrix and a synthetic
ill-conditioned matrix. Write iteration histories and summary CSV files to
--output-dir; use plot_ENR_ISTA.py to generate the residual figures.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, replace
from time import time

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optim_algo import AndersonAcc, GlobalAndersonSolver


DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_M_VALUES = (1, 3, 5, 10, 20, 30)
HISTORY_FILENAME = "enr_ista_history.csv"
SUMMARY_FILENAME = "enr_ista_summary.csv"
ALGORITHM_COLUMN_WIDTH = 36
SUMMARY_TABLE_WIDTH = 122 + (ALGORITHM_COLUMN_WIDTH - 28)

ALGORITHMS = (
    "GlobalAndersonSolver eta=0",
    "GlobalAndersonSolver regularized",
    "AA_pure",
    "AA_residual",
    "Picard",
    "KM lambda=0.5",
)

HISTORY_FIELDS = (
    "case",
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
    "case",
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
class ENRCase:
    name: str
    display_name: str
    matrix_type: str
    beta: float
    alpha_factor: float
    gamma_kind: str
    mu_scale: float
    lambda_: float
    tol: float
    max_iters: int
    regularized_eta: float
    normalize_columns: bool
    spectral_method: str = "power"
    sparsity: float = 0.1
    noise_level: float = 0.1
    seed: int = 0
    num_rows: int = 1000
    num_cols: int = 2000
    condition_number: float = 1e6
    power_iters: int = 80
    spectral_margin: float = 1e-3

    @property
    def func_eval_max(self):
        return 2 * self.max_iters + 2


@dataclass(frozen=True)
class SolverConfig:
    max_iters: int
    func_eval_max: int
    tol: float
    lambda_: float
    regularized_eta: float
    gamma_k: object
    show: bool


CASES = (
    ENRCase(
        name="contractive_madelon",
        display_name="Contractive Madelon",
        matrix_type="madelon",
        beta=0.5,
        alpha_factor=1.8,
        gamma_kind="offset1",
        mu_scale=1e-5,
        lambda_=1.0,
        tol=1e-10,
        max_iters=2000,
        regularized_eta=1e-8,
        normalize_columns=False,
        spectral_method="exact",
    ),
    ENRCase(
        name="nonexpansive_ill_conditioned",
        display_name="Nonexpansive ill-conditioned",
        matrix_type="ill-conditioned",
        beta=1,
        alpha_factor=2.0,
        gamma_kind="one_minus_tol",
        mu_scale=1e-3,
        lambda_=0.5,
        tol=1e-10,
        max_iters=25000,
        regularized_eta=1e-10,
        normalize_columns=True,
        spectral_method="exact",
        num_rows=1000,
        num_cols=2000,
        condition_number=1e6,
    ),
)


def parse_number_list(value, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def soft_threshold(x, threshold):
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)


def spectral_norm_squared(A, num_iters):
    v = torch.randn(A.shape[1], dtype=A.dtype, device=A.device)
    v_norm = v.norm()
    if v_norm == 0:
        return torch.zeros((), dtype=A.dtype, device=A.device)
    v = v / v_norm

    for _ in range(num_iters):
        v = A.T @ (A @ v)
        v_norm = v.norm()
        if v_norm == 0:
            return torch.zeros((), dtype=A.dtype, device=A.device)
        v = v / v_norm

    Av = A @ v
    return torch.dot(Av, Av)


def largest_eigenvalue_ata(A, method, num_iters, power_margin):
    if method == "exact":
        sigma_max = torch.linalg.svdvals(A)[0]
        return (1.0 + 1e-12) * sigma_max.square()
    if method == "power":
        estimate = spectral_norm_squared(A, num_iters)
        return (1.0 + power_margin) * estimate
    raise ValueError(f"Unsupported spectral_method: {method}")


def load_madelon_matrix(dtype, device):
    data_path = os.path.join(
        REPO_ROOT, "datasets", "ISTA", "madelon_train.data.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Madelon data file not found: {data_path}")

    data = np.loadtxt(data_path)
    A = torch.as_tensor(data, dtype=dtype, device=device)
    A = A - A.mean(dim=0, keepdim=True)
    column_stds = torch.clamp(
        A.std(dim=0, keepdim=True, unbiased=False),
        min=torch.finfo(dtype).tiny)
    return A / column_stds


def make_ill_conditioned_matrix(case, dtype, device):
    if case.condition_number < 1:
        raise ValueError("condition_number must be at least 1.")

    rank = min(case.num_rows, case.num_cols)
    left_factors = torch.randn(case.num_rows, rank, dtype=dtype, device=device)
    right_factors = torch.randn(case.num_cols, rank, dtype=dtype, device=device)
    U, _ = torch.linalg.qr(left_factors, mode="reduced")
    V, _ = torch.linalg.qr(right_factors, mode="reduced")

    sigma_max = math.sqrt(case.num_rows) + math.sqrt(case.num_cols)
    singular_values = torch.logspace(
        0.0,
        -math.log10(case.condition_number),
        rank,
        dtype=dtype,
        device=device,
    )
    singular_values = singular_values * sigma_max
    return (U * singular_values.reshape(1, -1)) @ V.T


def make_design_matrix(case, dtype, device):
    if case.matrix_type == "madelon":
        return load_madelon_matrix(dtype, device)
    if case.matrix_type == "ill-conditioned":
        return make_ill_conditioned_matrix(case, dtype, device)
    raise ValueError(f"Unsupported matrix_type: {case.matrix_type}")


def make_gamma(case):
    if case.gamma_kind == "offset1":
        return lambda k, x, Gk, Fk, m_k, kappa: (k + 1) / (k + 2)
    if case.gamma_kind == "paper_window":
        return "paper_window"
    if case.gamma_kind == "one_minus_tol":
        # return 1 - case.tol
        return 1 - 1e-5
    raise ValueError(f"Unsupported gamma_kind: {case.gamma_kind}")


def make_problem(case, dtype, device):
    torch.manual_seed(case.seed)
    A = make_design_matrix(case, dtype, device)
    if case.normalize_columns:
        column_norms = torch.clamp(
            A.norm(dim=0, keepdim=True), min=torch.finfo(dtype).tiny)
        A = A / column_norms

    AT = A.T
    num_rows, num_cols = A.shape

    def Amap(x):
        return A @ x

    def ATmap(y):
        return AT @ y

    x_true = torch.zeros(num_cols, dtype=dtype, device=device)
    support = torch.rand(num_cols, device=device) < case.sparsity
    x_true[support] = torch.randn(int(support.sum()), dtype=dtype, device=device)
    noise = torch.randn(num_rows, dtype=dtype, device=device)
    b = Amap(x_true) + case.noise_level * noise
    Atb = ATmap(b)

    x0 = torch.zeros(num_cols, dtype=dtype, device=device)
    mu_max = torch.norm(Atb, p=float("inf"))
    mu = case.mu_scale * mu_max

    lmax = largest_eigenvalue_ata(
        A,
        case.spectral_method,
        case.power_iters,
        case.spectral_margin,
    )
    smooth_L = lmax + mu * (1 - case.beta)
    alpha = case.alpha_factor / smooth_L
    threshold = alpha * mu * case.beta

    min_smooth_eig = mu * (1 - case.beta)
    contraction_bound = torch.maximum(
        torch.abs(1 - alpha * min_smooth_eig),
        torch.abs(1 - alpha * smooth_L),
    )

    def ista_map(x):
        grad_smooth = ATmap(Amap(x)) - Atb + mu * (1 - case.beta) * x
        return soft_threshold(x - alpha * grad_smooth, threshold)

    info = {
        "rows": num_rows,
        "cols": num_cols,
        "beta": case.beta,
        "mu": mu.item(),
        "mu_max": mu_max.item(),
        "L": smooth_L.item(),
        "alpha": alpha.item(),
        "threshold": threshold.item(),
        "contraction_bound": contraction_bound.item(),
        "normalize_columns": case.normalize_columns,
        "spectral_method": case.spectral_method,
    }
    return ista_map, x0, info


def solver_config_for_case(case, show):
    return SolverConfig(
        max_iters=case.max_iters,
        func_eval_max=case.func_eval_max,
        tol=case.tol,
        lambda_=case.lambda_,
        regularized_eta=case.regularized_eta,
        gamma_k=make_gamma(case),
        show=show,
    )


def config_with_max_iters(config, max_iters):
    return replace(config, max_iters=max_iters, func_eval_max=2 * max_iters + 2)


def run_global_anderson_solver(ista_map, x0, anderson_m, config, eta):
    solver = GlobalAndersonSolver(
        ista_map,
        x0.clone(),
        m=anderson_m,
        lambda_=config.lambda_,
        mainLoopMaxItrs=config.max_iters,
        funcEvalMax=config.func_eval_max,
        gamma_k=config.gamma_k,
        residualTol=config.tol,
        eta=eta,
        show=config.show,
        print_final=False,
    )
    return solver.run()


def run_pure_aa(ista_map, x0, anderson_m, config):
    x, record = AndersonAcc(
        ista_map,
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
    )
    return x, record


def run_residual_aa(ista_map, x0, anderson_m, config):
    x, record = AndersonAcc(
        ista_map,
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
        arg="residual",
        restart=False,
        regularization_eta=config.regularized_eta,
        print_final=False,
    )
    return x, record


def run_base_iteration(ista_map, x0, config, lambda_value):
    x = x0.clone()
    tmk = 0.0
    orcl = 0
    iters = 0

    Sx = ista_map(x).detach()
    orcl += 1
    Fx = Sx - x
    residual_norm = Fx.norm()
    residual0_norm = torch.clamp(
        residual_norm.detach().clone(), min=torch.finfo(x.dtype).tiny)
    record = torch.tensor(
        [residual_norm / residual0_norm, residual_norm, orcl, tmk, 0, 1],
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, -1)

    while residual_norm / residual0_norm >= config.tol \
            and iters < config.max_iters \
            and orcl < config.func_eval_max:
        t0 = time()
        x = (1 - lambda_value) * x + lambda_value * Sx
        Sx = ista_map(x).detach()
        orcl += 1
        Fx = Sx - x
        residual_norm = Fx.norm()
        iters += 1
        tmk += time() - t0
        row = torch.tensor(
            [residual_norm / residual0_norm, residual_norm, orcl, tmk, 0, 1],
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
    if algorithm.startswith("KM lambda="):
        return "KM"
    return "AA" if direction_value == 0 else "Picard"


def effective_m_for_row(algorithm, row_values, iter_index, anderson_m):
    if algorithm.startswith("GlobalAndersonSolver"):
        return int(round(row_values[7]))
    if algorithm in ("AA_pure", "AA_residual"):
        return min(iter_index, anderson_m)
    return 0


def record_to_history_rows(case_name, algorithm, anderson_m, record):
    values = record.detach().cpu().numpy()
    rows = []
    for iter_index, row in enumerate(values):
        rows.append({
            "case": case_name,
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


def summarize_history_rows(case_name, algorithm, anderson_m, rows):
    final = rows[-1]
    iterations = int(final["iter"])
    aa_steps = sum(
        1 for row in rows if row["iter"] > 0 and row["step_type"] == "AA")
    aa_rate_percent = 100.0 * aa_steps / max(iterations, 1)
    return {
        "case": case_name,
        "algorithm": algorithm,
        "m": anderson_m,
        "iterations": iterations,
        "time": final["time"],
        "final_rel_res": final["rel_res"],
        "final_residual": final["residual"],
        "aa_steps": aa_steps,
        "aa_rate_percent": aa_rate_percent,
    }


def print_summary_header(case, anderson_m):
    print("-" * SUMMARY_TABLE_WIDTH)
    print(f"case = {case.name}, m = {anderson_m}")
    print(
        f"  {'algorithm':<{ALGORITHM_COLUMN_WIDTH}} {'iterations':>10} "
        f"{'time(s)':>12} "
        f"{'rel_res':>12} {'residual':>12} {'AA_steps':>9} {'AA_rate%':>10}"
    )
    print("-" * SUMMARY_TABLE_WIDTH)


def print_summary_row(summary):
    print(
        f"  {summary['algorithm']:<{ALGORITHM_COLUMN_WIDTH}} "
        f"{summary['iterations']:>10d} "
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


def run_and_record(case_name, algorithm, run_callable, anderson_m,
                   history_rows, summary_rows):
    _, record = run_callable()
    rows = record_to_history_rows(case_name, algorithm, anderson_m, record)
    summary = summarize_history_rows(case_name, algorithm, anderson_m, rows)
    history_rows.extend(rows)
    summary_rows.append(summary)
    print_summary_row(summary)
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal ENR/ISTA experiment for Anderson acceleration.")
    parser.add_argument("--m-values", default="1,3,5,10,20,30",
                        help="Comma-separated Anderson memory values.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run both cases with m=1 and max_iters=5 for a quick check.")
    args = parser.parse_args()

    args.m_values = parse_number_list(args.m_values, int)
    if not args.m_values:
        raise ValueError("--m-values must contain at least one integer.")
    for anderson_m in args.m_values:
        if anderson_m <= 0:
            raise ValueError("All m values must be positive integers.")
    return args


def active_cases(smoke):
    if not smoke:
        return CASES
    return tuple(replace(case, max_iters=5) for case in CASES)


def main():
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    m_values = (1,) if args.smoke else args.m_values

    history_rows = []
    summary_rows = []

    print("ENR/ISTA formal experiment")
    print(f"  algorithms={', '.join(ALGORITHMS)}")
    print(f"  m values={m_values}")
    print(f"  dtype={args.dtype}, device={args.device}")
    if args.smoke:
        print("  smoke mode=True, max_iters=5")

    for case in active_cases(args.smoke):
        ista_map, x0, info = make_problem(case, dtype, device)
        config = solver_config_for_case(case, args.show)
        if args.smoke:
            config = config_with_max_iters(config, case.max_iters)

        print("=" * SUMMARY_TABLE_WIDTH)
        print(f"case={case.name} ({case.display_name})")
        print(f"  matrix_type={case.matrix_type}, rows={info['rows']}, cols={info['cols']}")
        print(f"  sparsity={case.sparsity}, noise_level={case.noise_level}")
        print(f"  beta={case.beta}, mu_scale={case.mu_scale:.1e}, mu={info['mu']:.4e}")
        print(f"  alpha={info['alpha']:.4e}, L={info['L']:.4e}, "
              f"threshold={info['threshold']:.4e}, "
              f"spectral_method={info['spectral_method']}")
        print(f"  normalize_columns={case.normalize_columns}, lambda={case.lambda_:.4e}")
        print(f"  gamma_kind={case.gamma_kind}, "
              f"regularized_eta={case.regularized_eta:.4e}")
        print(f"  tol={case.tol:.1e}, max_iters={config.max_iters}, "
              f"func_eval_max={config.func_eval_max}")
        print(f"  estimated contraction bound={info['contraction_bound']:.12f}")

        base_records = {}
        regularized_global_label = (
            f"GlobalAndersonSolver eta={case.regularized_eta:g}"
        )
        for anderson_m in m_values:
            print_summary_header(case, anderson_m)
            run_and_record(
                case.name,
                "GlobalAndersonSolver eta=0",
                lambda m=anderson_m: run_global_anderson_solver(
                    ista_map, x0, m, config, eta=0.0),
                anderson_m,
                history_rows,
                summary_rows,
            )
            run_and_record(
                case.name,
                regularized_global_label,
                lambda m=anderson_m: run_global_anderson_solver(
                    ista_map, x0, m, config,
                    eta=config.regularized_eta),
                anderson_m,
                history_rows,
                summary_rows,
            )
            run_and_record(
                case.name,
                "AA_pure",
                lambda m=anderson_m: run_pure_aa(
                    ista_map, x0, m, config),
                anderson_m,
                history_rows,
                summary_rows,
            )
            run_and_record(
                case.name,
                "AA_residual",
                lambda m=anderson_m: run_residual_aa(
                    ista_map, x0, m, config),
                anderson_m,
                history_rows,
                summary_rows,
            )
            for base_label, lambda_value in (
                    ("Picard", 1.0),
                    ("KM lambda=0.5", 0.5)):
                if base_label not in base_records:
                    _, base_records[base_label] = run_base_iteration(
                        ista_map, x0, config, lambda_value)
                base_rows = record_to_history_rows(
                    case.name, base_label, anderson_m,
                    base_records[base_label])
                base_summary = summarize_history_rows(
                    case.name, base_label, anderson_m, base_rows)
                history_rows.extend(base_rows)
                summary_rows.append(base_summary)
                print_summary_row(base_summary)

    history_path = os.path.join(args.output_dir, HISTORY_FILENAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILENAME)
    write_csv(history_path, history_rows, HISTORY_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"Saved history CSV to {history_path}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
