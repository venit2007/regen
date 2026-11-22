#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-$repo_dir/venv/bin/python}"
read -r -a deltas <<< "${DELTA_VALUES:-0.0005 0.0007 0.0009 0.0011 0.0013 0.0015 0.0017 0.0019}"
read -r -a rib_widths <<< "${W_B_VALUES:-0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0010 0.0012 0.0014}"
if [[ ! -v 'deltas[0]' || ! -v 'rib_widths[0]' ]]; then
    printf 'Supply space-separated values in metres for DELTA_VALUES and W_B_VALUES.\n' >&2
    exit 2
fi

if (( $# > 0 )); then
    # Require a fresh directory so an earlier sweep cannot be overwritten.
    mkdir -- "$1"
    output_dir="$(cd -- "$1" && pwd)"
else
    output_dir="$(mktemp -d "$PWD/cooling-sweep.XXXXXX")"
fi
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT
cp -- "$repo_dir"/{solver.py,channel.py,plot.py,config.json} "$work_dir/"
mv -- "$work_dir/config.json" "$work_dir/base_config.json"
mkdir -- "$work_dir/data"
cp -- "$repo_dir/data/HORUS_regen_cooling_axial_inputs.csv" "$work_dir/axial_input.csv"
cp -r -- "$repo_dir/n2o_property_tables" "$work_dir/"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$work_dir/matplotlib"

cat > "$work_dir/sweep_case.py" <<'PY'
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
from pathlib import Path
import runpy
from shutil import copyfile
import sys
import traceback

WORK = Path(__file__).resolve().parent
THRESHOLD_M = 0.28


def run_case(output, case_id, delta, rib_width):
    import channel
    import plot

    # The sweep only needs the final 3D graph. Suppress per-run figures.
    for name in vars(plot):
        if name.startswith("plot_") or name in ("refresh_live_plots", "show_plots"):
            setattr(plot, name, lambda *args, **kwargs: None)

    mesh_path = WORK / "data" / "HORUS_regen_cooling_axial_inputs.csv"
    copyfile(WORK / "axial_input.csv", mesh_path)
    config = json.loads((WORK / "base_config.json").read_text())
    log_path = output / f"case-{case_id}.log"
    n_solved, last_x, status = "", "", "ERROR"
    with log_path.open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                config.update(delta=float(delta), w_b=float(rib_width))
                (WORK / "config.json").write_text(json.dumps(config), encoding="utf-8")
                # The source mesh is in increasing x. Coolant marches in reverse.
                with mesh_path.open(newline="", encoding="utf-8") as source:
                    x = [float(row["x_m"]) for row in csv.DictReader(source)][::-1]
                x_cells = [(upstream + downstream) / 2 for upstream, downstream in zip(x, x[1:])]
                try:
                    sys.argv = [str(WORK / "solver.py")]
                    runpy.run_path(sys.argv[0], run_name="__main__")
                    n_solved = len(x_cells)
                except channel.IncompleteThermalSolve as exc:
                    n_solved = exc.partial_results["n_solved_cells"]
                    print(exc)
                if n_solved:
                    last_x = float(x_cells[n_solved - 1])
                status = "PASS" if n_solved and last_x < THRESHOLD_M else "FAIL"
            except Exception:
                traceback.print_exc()

    results_path = output / "results.csv"
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as results:
        writer = csv.writer(results)
        if write_header:
            writer.writerow(("delta_m", "w_b_m", "cells_solved", "last_solved_cell_x_m", "status", "log"))
        writer.writerow((delta, rib_width, n_solved, last_x, status, log_path.name))
    count = f"{n_solved} cells" if n_solved != "" else "unknown cell count"
    print(f"delta={delta} m, w_b={rib_width} m: {status} ({count})")


def plot_results(output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with (output / "results.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    figure = plt.figure(figsize=(9, 7), layout="constrained")
    axes = figure.add_subplot(projection="3d")
    for status, color, marker in (("PASS", "tab:green", "o"), ("FAIL", "tab:red", "X")):
        cases = [row for row in rows if row["status"] == status]
        if cases:
            axes.scatter(
                [float(row["delta_m"]) * 1e3 for row in cases],
                [float(row["w_b_m"]) * 1e3 for row in cases],
                [int(row["cells_solved"]) for row in cases],
                color=color, marker=marker, s=60, label=status,
            )
    axes.set(
        xlabel="Channel height δ [mm]", ylabel="Rib thickness w_b [mm]",
        zlabel="Cells solved", title=f"Cooling sweep · pass if last solved cell is below x = {THRESHOLD_M:g} m",
    )
    axes.zaxis.set_major_locator(MaxNLocator(integer=True))
    if axes.has_data():
        axes.legend()
    errors = sum(row["status"] == "ERROR" for row in rows)
    if errors:
        figure.text(0.03, 0.02, f"{errors} cases with unknown cell count. See case logs.")
    figure.savefig(output / "cells_solved_3d.png", dpi=170)


mode, output_dir, *arguments = sys.argv[1:]
if mode == "case":
    run_case(Path(output_dir), *arguments)
else:
    plot_results(Path(output_dir))
PY

case_id=0
for delta in "${deltas[@]}"; do
    for w_b in "${rib_widths[@]}"; do
        case_id=$((case_id + 1))
        "$python_bin" -B "$work_dir/sweep_case.py" case "$output_dir" "$case_id" "$delta" "$w_b"
    done
done

"$python_bin" -B "$work_dir/sweep_case.py" plot "$output_dir"
printf 'Sweep results and 3D graph saved in %s\n' "$output_dir"
