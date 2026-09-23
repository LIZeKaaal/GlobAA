"""
Run the Section 3.1 gamma_k sweep for the AA-GD counterexample.

This script generates the data for Table 1. It compares several admissible
GlobalAndersonSolver gamma_k schedules against Picard (GD), AA_pure, and
AA_residual for two constant initial points and one normal random initial
point (seed 42), with m in {1, 3, 5}. AA_residual uses the AndersonAcc default
regularization parameter; --eta controls AA_pure and GlobalAndersonSolver.
"""

import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optim_algo import AndersonAcc, GlobalAndersonSolver  # noqa: E402
from run_counterexample import (  # noqa: E402
    DEFAULT_DIM,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOL,
    HighDimNonseparableCounterexample,
    IterateRecorder,
    aa_step_percentage,
    compute_diagnostics,
    run_picard,
    run_pure_aa,
    step_type_from_record,
)


DEFAULT_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "gamma_sweep_results.csv")
SWEEP_MAX_ITERS = 500
SWEEP_FUNC_EVAL_MAX = 4 * SWEEP_MAX_ITERS + 4
RANDN_SEED = 42
INITIAL_CASES = (
    {
        "name": "t0=2.1",
        "display": "u0=2.1*1",
        "kind": "constant",
        "t0": 2.1,
        "seed": None,
    },
    {
        "name": "t0=300",
        "display": "u0=300*1",
        "kind": "constant",
        "t0": 300.0,
        "seed": None,
    },
    {
        "name": f"randn_seed_{RANDN_SEED}",
        "display": f"u0=randn(seed={RANDN_SEED})",
        "kind": "randn",
        "t0": None,
        "seed": RANDN_SEED,
    },
)
M_VALUES = (1, 3, 5)

GAMMA_SCHEDULES = (
    {
        "name": "one_minus_one_over_k_plus_2",
        "label": "1-1/(k+2)",
        "formula": "1 - 1 / (k + 2)",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: 1 - 1 / (k + 2),
    },
    {
        "name": "one_minus_one_over_k_plus_301",
        "label": "1-1/(k+301)",
        "formula": "1 - 1 / (k + 301)",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: 1 - 1 / (k + 301),
    },
    {
        "name": "one_minus_one_over_5k_plus_2",
        "label": "1-1/(5k+2)",
        "formula": "1 - 1 / (5 * k + 2)",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: (
            1 - 1 / (5 * k + 2)),
    },
    {
        "name": "one_minus_one_over_100k_plus_2",
        "label": "1-1/(100k+2)",
        "formula": "1 - 1 / (100 * k + 2)",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: (
            1 - 1 / (100 * k + 2)),
    },
    {
        "name": "one_minus_one_over_sqrt_k_plus_2",
        "label": "1-1/sqrt(k+2)",
        "formula": "1 - 1 / sqrt(k + 2)",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: (
            1 - 1 / math.sqrt(k + 2)),
    },
    {
        "name": "one_minus_one_over_10_log_k_plus_2",
        "label": "1-1/(10log(k+2))",
        "formula": "1.0 - 1.0 / (10.0 * log(k + 2))",
        "fn": lambda k, x, Gk, Fk, m_k, kappa: (
            1.0 - 1.0 / (10.0 * math.log(k + 2))),
    },
)

CSV_FIELDS = (
    "case",
    "t0",
    "gamma_k_name",
    "gamma_k_formula",
    "algorithm",
    "m",
    "iter",
    "rel_res",
    "residual",
    "time",
    "step_type",
    "diagnostic_residual",
    "objective_value",
    "x_norm",
    "step_from_prev_norm",
    "s_value",
)


def make_solver_args(cli_args):
    return SimpleNamespace(
        eta=cli_args.eta,
        lambda_=DEFAULT_LAMBDA,
        max_iters=SWEEP_MAX_ITERS,
        func_eval_max=SWEEP_FUNC_EVAL_MAX,
        tol=DEFAULT_TOL,
        pure_aa_restart=False,
        residual_aa_restart=False,
        show=False,
    )


def run_global_with_gamma(gd_map, x0, anderson_m, args, gamma_k,
                          iterate_callback=None):
    solver = GlobalAndersonSolver(
        gd_map,
        x0.clone(),
        m=anderson_m,
        lambda_=args.lambda_,
        mainLoopMaxItrs=args.max_iters,
        funcEvalMax=args.func_eval_max,
        gamma_k=gamma_k,
        residualTol=args.tol,
        eta=args.eta,
        show=args.show,
        print_final=False,
        iterate_callback=iterate_callback,
    )
    return solver.run()


def run_residual_aa(gd_map, x0, anderson_m, args,
                    iterate_callback=None):
    return AndersonAcc(
        gd_map,
        x0.clone(),
        anderson_m,
        args.lambda_,
        1.0,
        args.max_iters,
        args.func_eval_max,
        0,
        0,
        0,
        0,
        0,
        gradTol=args.tol,
        show=args.show,
        arg="residual",
        restart=args.residual_aa_restart,
        print_final=False,
        iterate_callback=iterate_callback,
    )


def run_algorithm(problem, x0, anderson_m, algorithm, args, gamma_k=None):
    recorder = IterateRecorder()
    if algorithm == "Picard (GD)":
        _, record = run_picard(problem.map, x0, args, recorder)
    elif algorithm == "AA_pure":
        _, record = run_pure_aa(problem.map, x0, anderson_m, args, recorder)
    elif algorithm == "AA_residual":
        _, record = run_residual_aa(
            problem.map, x0, anderson_m, args, recorder)
    elif algorithm == "GlobalAndersonSolver":
        _, record = run_global_with_gamma(
            problem.map, x0, anderson_m, args, gamma_k, recorder)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    iterations, iterates = recorder.as_tensors()
    diagnostics = compute_diagnostics(problem, iterations, iterates)
    return record, diagnostics


def rows_for_algorithm(case_name, t0, gamma_name, gamma_formula, algorithm,
                       anderson_m, record, diagnostics):
    rows = []
    iterations = diagnostics["iterations"]
    step_norms = diagnostics["step_norms"]
    for row_index, row in enumerate(record.detach().cpu()):
        step_from_prev = (
            "" if row_index == 0 else f"{step_norms[row_index - 1].item():.17g}"
        )
        rows.append({
            "case": case_name,
            "t0": "" if t0 is None else f"{t0:.17g}",
            "gamma_k_name": gamma_name,
            "gamma_k_formula": gamma_formula,
            "algorithm": algorithm,
            "m": "" if anderson_m is None else int(anderson_m),
            "iter": int(iterations[row_index].item()),
            "rel_res": f"{row[0].item():.17g}",
            "residual": f"{row[1].item():.17g}",
            "time": f"{row[3].item():.17g}",
            "step_type": step_type_from_record(algorithm, record, row_index),
            "diagnostic_residual": (
                f"{diagnostics['residual_norms'][row_index].item():.17g}"),
            "objective_value": (
                f"{diagnostics['objective_values'][row_index].item():.17g}"),
            "x_norm": f"{diagnostics['x_norms'][row_index].item():.17g}",
            "step_from_prev_norm": step_from_prev,
            "s_value": f"{diagnostics['s_values'][row_index].item():.17g}",
        })
    return rows


def print_summary_header():
    print("-" * 150)
    print(
        f"  {'algorithm':<34} {'gamma_k':<24} {'iterations':>10} "
        f"{'rel_res':>12} {'residual':>12} {'AA step %':>10} "
        f"{'||u||':>12} {'s(u)':>12}"
    )
    print("-" * 150)


def print_summary_row(algorithm, anderson_m, gamma_label, record, diagnostics):
    if anderson_m is None:
        display_algorithm = algorithm
    else:
        display_algorithm = f"{algorithm}, m={anderson_m}"
    aa_pct = aa_step_percentage(algorithm, record)
    aa_pct_text = "-" if aa_pct is None else f"{aa_pct:9.2f}%"
    print(
        f"  {display_algorithm:<34} {gamma_label:<24} "
        f"{record.shape[0] - 1:>10d} "
        f"{record[-1, 0].item():>12.4e} {record[-1, 1].item():>12.4e} "
        f"{aa_pct_text:>10} "
        f"{diagnostics['x_norms'][-1].item():>12.4e} "
        f"{diagnostics['s_values'][-1].item():>12.4e}"
    )


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Section 3.1 gamma_k sweep experiment.")
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help=("Regularization eta for AA_pure and GlobalAndersonSolver. "
              "AA_residual uses the AndersonAcc default."))
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output CSV path.")
    parser.add_argument("--dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def print_common_header(cli_args):
    case_text = ", ".join(case["display"] for case in INITIAL_CASES)
    m_text = ", ".join(str(value) for value in M_VALUES)
    print("Section 3.1 gamma_k sweep")
    print(f"  dim={DEFAULT_DIM}, initial cases={case_text}, m values={m_text}, "
          f"gamma={DEFAULT_GAMMA:.6g}")
    print(f"  pureAA/GlobAA eta={cli_args.eta:.4e}, "
          "resAA eta=AndersonAcc default (1e-8), "
          f"max_iters={SWEEP_MAX_ITERS}, "
          f"tol={DEFAULT_TOL:.1e}")
    print("  gamma_k schedules:")
    for schedule in GAMMA_SCHEDULES:
        print(f"    {schedule['label']}: {schedule['formula']}")


def run_sweep(problem, solver_args):
    all_rows = []
    picard_cache = {}
    aa_cache = {}

    for case in INITIAL_CASES:
        x0 = problem.initial_point(
            case["t0"], kind=case["kind"], seed=case["seed"])
        case_name = case["name"]
        t0 = case["t0"]

        picard_record, picard_diagnostics = run_algorithm(
            problem, x0, None, "Picard (GD)", solver_args)
        picard_cache[case_name] = (picard_record, picard_diagnostics)
        all_rows.extend(rows_for_algorithm(
            case_name, t0, "not_applicable", "", "Picard (GD)", None,
            picard_record, picard_diagnostics))

        for anderson_m in M_VALUES:
            print(f"\nCase: {case['display']}, m={anderson_m}")
            print_summary_header()

            print_summary_row(
                "Picard (GD)", None, "-", picard_record, picard_diagnostics)

            for algorithm in ("AA_pure", "AA_residual"):
                record, diagnostics = run_algorithm(
                    problem, x0, anderson_m, algorithm, solver_args)
                aa_cache[(case_name, anderson_m, algorithm)] = (
                    record, diagnostics)
                all_rows.extend(rows_for_algorithm(
                    case_name, t0, "not_applicable", "", algorithm,
                    anderson_m, record, diagnostics))
                print_summary_row(
                    algorithm, anderson_m, "-", record, diagnostics)

            for schedule in GAMMA_SCHEDULES:
                record, diagnostics = run_algorithm(
                    problem, x0, anderson_m, "GlobalAndersonSolver",
                    solver_args, gamma_k=schedule["fn"])
                all_rows.extend(rows_for_algorithm(
                    case_name, t0, schedule["name"], schedule["formula"],
                    "GlobalAndersonSolver", anderson_m, record, diagnostics))
                print_summary_row(
                    "GlobalAndersonSolver", anderson_m, schedule["label"],
                    record, diagnostics)

            print("-" * 150)

    return all_rows, picard_cache, aa_cache


def main():
    cli_args = parse_args()
    dtype = torch.float64 if cli_args.dtype == "float64" else torch.float32
    device = torch.device(cli_args.device)
    solver_args = make_solver_args(cli_args)
    problem = HighDimNonseparableCounterexample(
        dim=DEFAULT_DIM,
        gamma=DEFAULT_GAMMA,
        dtype=dtype,
        device=device,
    )

    print_common_header(cli_args)
    rows, _, _ = run_sweep(problem, solver_args)
    write_csv(rows, cli_args.output)
    print(f"\nSaved CSV to {cli_args.output}")


if __name__ == "__main__":
    main()
