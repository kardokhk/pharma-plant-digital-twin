"""Level 2 experiment: compare three fluid-bed dryer end-point policies.

Policies, each with one parameter tuned on a calibration set and evaluated on a held-out test set:
  fixed   : dry for a fixed time (validated recipe time)
  temp    : stop when product temperature reaches a set value (common practice)
  twin    : stop when the estimator's upper 95% bound on LOD falls below a set value

Two test sets: "stable" (same climate as calibration) and "humid" (inlet wet-bulb temperature
5 degC higher, as in a humid summer), with every policy parameter frozen from calibration.
Also measured: how well the twin predicts the time to reach 2% LOD, made at the moment its estimate
first crosses 6% LOD, which is what the plant twin needs for scheduling.

Estimator settings were chosen on a separate development set (seed 1) before this script ran.
Outputs: results/dryer/summary.csv, results/dryer/test_batches.csv, results/dryer/cycle_samples.json,
         results/dryer/example_trace.csv
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin import dryer as d  # noqa: E402
from plant_twin.stats import wilson, boot_ci  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dryer")
SEED_CAL, SEED_TEST, SEED_HUMID = 1001, 2002, 3003
N_CAL, N_TEST = 500, 1000

GRID = {
    "fixed": np.arange(40.0, 110.01, 0.5),
    "temp": np.arange(38.0, 59.81, 0.1),
    "twin": np.arange(0.016, 0.0301, 0.0002),
}


def one_batch(args):
    seed, i, t_wb_mean = args
    rng = np.random.default_rng([seed, i])
    b = d.sample_batch(rng, t_wb_mean=t_wb_mean)
    t, x, ytp, yr = d.simulate_truth(b, rng)
    zs, sd = d.run_estimator_states(b, ytp, yr)
    xh = zs[:, 0]
    # end-time prediction at the first time the estimate crosses 6% LOD
    j = int(np.argmax(d.lod(xh) <= 0.06))
    pred = t[j] + d.predict_end_time(zs[j], b.t_wb_meas, b.t_in)
    actual = t[int(np.argmax(d.lod(x) <= d.LOD_TARGET))]
    b.pred_info = (t[j], pred, actual)
    return b, t, x, ytp, xh, sd


def evaluate(batches, policy, p):
    rows = []
    for b, t, x, ytp, xh, sd in batches:
        if policy == "fixed":
            i = d.stop_fixed_time(t, p)
        elif policy == "temp":
            i = d.stop_product_temp(ytp, p)
        else:
            i = d.stop_twin(xh, sd, p)
        rows.append(d.batch_outcome(t, x, i))
    return pd.DataFrame(rows)


def score(df):
    in_spec = ((df.lod_first >= d.LOD_LOW) & (df.lod_first <= d.LOD_HIGH)).mean()
    return in_spec, -df.cycle_min.mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        cal = pool.map(one_batch, [(SEED_CAL, i, 27.0) for i in range(N_CAL)], chunksize=4)
        stable = pool.map(one_batch, [(SEED_TEST, i, 27.0) for i in range(N_TEST)], chunksize=4)
        humid = pool.map(one_batch, [(SEED_HUMID, i, 32.0) for i in range(N_TEST)], chunksize=4)

    rows, samples, per_batch = [], {}, []
    best_par = {pol: max(grid, key=lambda p: score(evaluate(cal, pol, p))) for pol, grid in GRID.items()}
    for (testset, test), policy in [((ts, tt), pol) for ts, tt in (("stable", stable), ("humid", humid))
                                    for pol in GRID]:
        best = best_par[policy]
        df = evaluate(test, policy, best)
        df["policy"] = policy
        df["test_set"] = testset
        per_batch.append(df)
        n = len(df)
        in_spec = int(((df.lod_first >= d.LOD_LOW) & (df.lod_first <= d.LOD_HIGH)).sum())
        under = int(df.under_first.sum())
        over = int(df.over_dried.sum())
        med, lo, hi = boot_ci(df.cycle_min.values, np.median, seed=7)
        mean, mlo, mhi = boot_ci(df.cycle_min.values, np.mean, seed=7)
        rows.append({
            "test_set": testset, "policy": policy, "parameter": round(float(best), 4), "n_test": n,
            "in_spec_first": in_spec, "in_spec_pct": 100 * in_spec / n,
            "in_spec_ci": wilson(in_spec, n),
            "under_dried_first": under, "under_ci": wilson(under, n),
            "over_dried": over, "over_ci": wilson(over, n),
            "cycle_median_min": med, "cycle_median_ci": (lo, hi),
            "cycle_mean_min": mean, "cycle_mean_ci": (mlo, mhi),
            "cycle_p10": float(np.percentile(df.cycle_min, 10)),
            "cycle_p90": float(np.percentile(df.cycle_min, 90)),
        })
        if testset == "stable":
            samples[policy] = [round(v, 2) for v in df.cycle_min.tolist()]

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.concat(per_batch).to_csv(os.path.join(OUT, "test_batches.csv"), index=False)
    with open(os.path.join(OUT, "cycle_samples.json"), "w") as fh:
        json.dump(samples, fh)

    pred = pd.DataFrame([dict(zip(("t_pred_min", "pred_end_min", "actual_end_min"), bt[0].pred_info),
                              test_set=name) for name, ts in (("stable", stable), ("humid", humid)) for bt in ts])
    pred["error_min"] = pred.pred_end_min - pred.actual_end_min
    naive = float(np.median([bt[0].pred_info[2] for bt in cal]))   # calibration-set median end time
    pred["naive_error_min"] = naive - pred.actual_end_min
    pred["horizon_min"] = pred.actual_end_min - pred.t_pred_min
    pred.to_csv(os.path.join(OUT, "endtime_prediction.csv"), index=False)
    print(pred.groupby("test_set")[["error_min", "horizon_min"]].describe().T.to_string())

    # one example trace for the figure (stable test batch 3)
    b, t, x, ytp, xh, sd = stable[3]
    pd.DataFrame({"t_min": t, "lod_true": d.lod(x), "lod_est": d.lod(xh),
                  "lod_upper95": d.lod(xh + 1.645 * sd), "lod_lower95": d.lod(np.maximum(xh - 1.645 * sd, d.X_E)),
                  "tp_meas": ytp}).to_csv(os.path.join(OUT, "example_trace.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
