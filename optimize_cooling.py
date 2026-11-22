"""Search channel height and rib thickness with NSGA-II."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
from pathlib import Path
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory, mkdtemp
import traceback


ROOT = Path(__file__).resolve().parent
MESH = Path("data/HORUS_regen_cooling_axial_inputs.csv")
THROAT_X_M = 0.28
POPULATION = 100
OFFSPRING = 50
MUTATION_PROBABILITY = 0.2
GENERATIONS = 40
FIELDS = (
    "evaluation", "delta_m", "w_b_m", "cells_solved", "last_cell_x_m",
    "pressure_drop_pa", "thermal_converged", "status", "log",
)


class CoolingEvaluator:
    def __init__(self, work, output):
        self.work = work
        self.output = output
        self.config = json.loads((ROOT / "config.json").read_text())
        self.records = {}
        for name in ("solver.py", "channel.py", "plot.py"):
            shutil.copy2(ROOT / name, work / name)
        shutil.copytree(ROOT / "n2o_property_tables", work / "n2o_property_tables")
        (work / MESH).parent.mkdir()
        self.mesh = (ROOT / MESH).read_bytes()
        with (ROOT / MESH).open(newline="", encoding="utf-8") as source:
            # Coolant flows from larger to smaller x in this solver.
            x = [float(row["x_m"]) for row in csv.DictReader(source)][::-1]
        self.x_cells = [(a + b) / 2 for a, b in zip(x, x[1:])]
        self.required_cells = next(i for i, x_cell in enumerate(self.x_cells, 1) if x_cell < THROAT_X_M)

        sys.path.insert(0, str(work))
        import channel
        import plot

        self.incomplete = channel.IncompleteThermalSolve
        # Only the optimization plots are needed. Solver CSV writes stay in the scratch copy.
        for name in vars(plot):
            if name.startswith("plot_") or name in ("refresh_live_plots", "show_plots"):
                setattr(plot, name, lambda *args, **kwargs: None)

        with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, FIELDS).writeheader()

    def evaluate(self, design):
        delta, rib_width = key = tuple(map(float, design))
        if key in self.records:
            return self.records[key]

        evaluation = len(self.records) + 1
        log_path = self.output / f"case-{evaluation:04d}.log"
        config = {**self.config, "delta": delta, "w_b": rib_width}
        (self.work / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (self.work / MESH).write_bytes(self.mesh)
        cells, pressure_drop, converged = None, None, False

        solver_path = str(self.work / "solver.py")
        argv = sys.argv
        try:
            sys.argv = [solver_path]
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                try:
                    result = runpy.run_path(solver_path, run_name="__main__")["thermal"]
                    cells = len(self.x_cells)
                    converged = True
                except self.incomplete as exc:
                    result = exc.partial_results
                    cells = result["n_solved_cells"]
                    print(exc)
                except (RuntimeError, ValueError):
                    result = None
                    traceback.print_exc()
                if result is not None:
                    pressure_drop = float(result["p"][0] - result["p"][-1])
        finally:
            sys.argv = argv

        if cells is None:
            status = "ERROR"
        else:
            status = "PASS" if cells >= self.required_cells else "FAIL"
        record = dict(
            evaluation=evaluation, delta_m=delta, w_b_m=rib_width, cells_solved=cells,
            last_cell_x_m=self.x_cells[cells - 1] if cells else None,
            pressure_drop_pa=pressure_drop, thermal_converged=converged,
            status=status, log=log_path.name,
        )
        with (self.output / "results.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, FIELDS).writerow(record)
        self.records[key] = record
        return record


def save_plots(output, records, pareto):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 6), layout="constrained")
    geometry = figure.add_subplot(1, 2, 1, projection="3d")
    objectives = figure.add_subplot(1, 2, 2)
    for status, color, marker in (("PASS", "tab:green", "o"), ("FAIL", "tab:red", "x")):
        cases = [record for record in records if record["status"] == status]
        if not cases:
            continue
        geometry.scatter(
            [r["delta_m"] * 1e3 for r in cases], [r["w_b_m"] * 1e3 for r in cases],
            [r["cells_solved"] for r in cases], c=color, marker=marker, label=status,
        )
        objectives.scatter(
            [r["pressure_drop_pa"] / 1e6 for r in cases], [r["cells_solved"] for r in cases],
            c=color, marker=marker, alpha=0.5, label=status,
        )
    if pareto:
        objectives.scatter(
            [r["pressure_drop_pa"] / 1e6 for r in pareto], [r["cells_solved"] for r in pareto],
            facecolors="none", edgecolors="black", s=90, label="Pareto front",
        )
    geometry.set(xlabel="Channel height δ [mm]", ylabel="Rib thickness w_b [mm]",
                 zlabel="Cells solved", title="Candidate geometry")
    objectives.set(xlabel="Modeled pressure drop [MPa]", ylabel="Cells solved",
                   title="Objective trade-off")
    for axes in (geometry, objectives):
        if axes.has_data():
            axes.legend()
    if not pareto:
        objectives.set_title("Objective trade-off · no feasible Pareto front")
    objectives.grid(alpha=0.25)
    figure.savefig(output / "nsga2_results.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for results and case logs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    args = parser.parse_args()

    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
    except ModuleNotFoundError as exc:
        if exc.name == "pymoo":
            parser.exit(2, 'Install the NSGA-II dependency with python -m pip install "pymoo>=0.6.1,<0.7"\n')
        raise
    import matplotlib

    matplotlib.use("Agg")
    output = args.output or Path(mkdtemp(prefix="cooling-nsga2.", dir=Path.cwd()))
    if args.output is not None:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Writing optimization results to {output.resolve()}", flush=True)

    with TemporaryDirectory(prefix="arka-nsga2-") as directory:
        evaluator = CoolingEvaluator(Path(directory), output)

        class CoolingProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=2, n_obj=2, n_ieq_constr=1,
                    xl=[0.0005, 0.0002], xu=[0.0019, 0.0014],
                )

            def _evaluate(self, design, out, *args, **kwargs):
                record = evaluator.evaluate(design)
                cells = record["cells_solved"]
                if cells is None:
                    out["F"] = [0, 1e9]
                    out["G"] = [evaluator.required_cells + 1]
                else:
                    out["F"] = [-cells, record["pressure_drop_pa"]]
                    out["G"] = [evaluator.required_cells - cells]

        result = minimize(
            CoolingProblem(),
            NSGA2(
                pop_size=POPULATION,
                n_offsprings=OFFSPRING,
                mutation=PM(prob=MUTATION_PROBABILITY, at_least_once=True),
                eliminate_duplicates=True,
            ),
            ("n_gen", GENERATIONS), seed=args.seed, verbose=True,
        )

    pareto = []
    if result.X is not None:
        import numpy as np

        pareto = [evaluator.records[tuple(map(float, design))] for design in np.atleast_2d(result.X)]
    with (output / "pareto_front.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(pareto)
    save_plots(output, evaluator.records.values(), pareto)
    print(f"Recorded {len(evaluator.records)} evaluations and {len(pareto)} feasible Pareto designs.")


if __name__ == "__main__":
    main()
