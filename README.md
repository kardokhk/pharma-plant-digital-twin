# Pharmaceutical plant digital twin: levels 2, 5 and 6

A compact, reproducible demonstrator of three digital twin levels for a hypothetical oral solid dose
(OSD) manufacturing site, built to show how a plant twin and an optimisation twin are developed, tested
and run on HPC.

| Level | Module | What it does |
|---|---|---|
| L2 unit operation | `plant_twin/dryer.py` | Fluid-bed dryer model with an extended Kalman filter soft sensor; three end-point rules compared on held-out batches |
| L5 plant | `plant_twin/plant.py` | Discrete-event simulation (SimPy) of the whole site: sequence-dependent cleaning, breakdowns, press wear and preventive maintenance, shift calendars, QC laboratory, CONWIP release control, campaign sequencing, quarterly economics |
| L6 optimisation | `scripts/02_plant_sweep.py`, `scripts/03_optimise.py` | Brute-force sweep of 34,560 operating configurations x 30 replications on one HPC node; ML surrogate loop checked against the brute-force answer; verification with fresh paired seeds |
| L6 quantum readiness | `plant_twin/sequencing.py` | Campaign sequencing as an exact MILP (HiGHS) and as a QUBO solved by simulated annealing (a classical stand-in; no quantum hardware used) |

All plant data and parameters are invented and illustrative.

## Main results

- 1,036,800 simulated plant-quarters in 624 s on one 128-core node.
- Best plan versus a baseline plan: +EUR 1.47 million margin per quarter (95% CI 1.46 to 1.48; 200 fresh paired replications); service level 69.3% to 99.0%.
- Surrogate loop with 0.55% of the brute-force budget found the true best configuration in 12 of 20 repeats (largest shortfall EUR 16,150); random search with the same budget found it in 0 of 20.
- Dryer: the product-temperature end-point rule (99.7% of batches in specification) beat the estimator-based rule (94.5%); the estimator's end-time forecast beat the historical median (mean absolute error 8.7 versus 12.1 min). The dryer end-point rule moved plant margin by less than 1%, because drying is not the bottleneck.
- Sequencing: MILP matched exhaustive search on 80 of 80 instances; QUBO annealing found the optimum on all 20 five-product instances, with a median gap of 29.3% at 12 products.

Every number above is in `results/report_numbers.json`, written by `scripts/06_report_numbers.py`.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 13 tests, about 2 s
./run_all.sh                       # full pipeline; about 25 core-hours, so use many cores
```

On a Slurm cluster, `slurm/*.sbatch` run the same steps on one 128-core node (about 19 minutes in total); set `--account` and `PY` for your site.
`NPROC` sets the number of worker processes. `REPS` sets replications in the sweep (default 30).

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_dryer_twin.py` | `results/dryer/` |
| 2 | `scripts/02_plant_sweep.py` | `results/sweep/` (`runs.npz` is 41 MB and not committed; rerun to create it) |
| 3 | `scripts/03_optimise.py` | `results/optimisation/` |
| 4 | `scripts/04_sequencing_benchmark.py` | `results/sequencing/` |
| 5 | `scripts/05_figures.py` | `figures/` (PDF and 600 dpi PNG) |
| 6 | `scripts/06_report_numbers.py` | `results/report_numbers.json` |

Seeds: dryer calibration 1001, stable test 2002, humid test 3003, estimator development 1; plant replication r uses seed r (common random numbers across configurations); verification seeds 1000 to 1199; surrogate repeats 0 to 19; sequencing instances seeded by (size, index).

## Limitations

Synthetic site and data; a step that starts within a shift runs to completion; fixed quarterly demand; one site and one route; single-flip simulated annealing is a weak baseline for quantum methods. See the report for details.

## Licence

MIT. Copyright (c) 2026 Kardokh Kakabra.
