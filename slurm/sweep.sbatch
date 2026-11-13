#!/bin/bash -l
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=plant-twin-sweep
#SBATCH --output=logs/%x-%j.out
# One CPU node (128 cores): dryer experiment, then the full plant scenario sweep.
set -euo pipefail
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NPROC=128
cd "$SLURM_SUBMIT_DIR"
lscpu | grep -E "Model name|^CPU\(s\)" ; date
$PY scripts/01_dryer_twin.py
$PY scripts/02_plant_sweep.py
date
