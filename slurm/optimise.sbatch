#!/bin/bash -l
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=plant-twin-opt
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PY=${PY:-python}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=120
cd "$SLURM_SUBMIT_DIR"
date; $PY scripts/03_optimise.py; date
