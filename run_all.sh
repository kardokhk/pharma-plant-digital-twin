#!/bin/bash
# Full pipeline: L2 dryer experiment -> L5/L6 sweep -> optimisation -> sequencing -> figures -> numbers.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
$PY scripts/01_dryer_twin.py
$PY scripts/02_plant_sweep.py
$PY scripts/03_optimise.py
$PY scripts/04_sequencing_benchmark.py
$PY scripts/05_figures.py
$PY scripts/06_report_numbers.py
