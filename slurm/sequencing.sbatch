#!/bin/bash -l
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=plant-twin-seq
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=128
cd "$SLURM_SUBMIT_DIR"
date; $PY scripts/04_sequencing_benchmark.py; date
