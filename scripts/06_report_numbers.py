"""Collect every number quoted in the demo report into results/report_numbers.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin.stats import boot_ci  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
out = {}

# dryer
sm = pd.read_csv(os.path.join(RES, "dryer", "summary.csv"))
out["dryer"] = {f"{r.test_set}_{r.policy}": {
    "parameter": r.parameter, "n": int(r.n_test), "in_spec": int(r.in_spec_first), "in_spec_pct": r.in_spec_pct,
    "in_spec_ci": r.in_spec_ci, "under": int(r.under_dried_first), "over": int(r.over_dried),
    "cycle_median": round(r.cycle_median_min, 1), "cycle_median_ci": r.cycle_median_ci,
    "cycle_p10": round(r.cycle_p10, 1), "cycle_p90": round(r.cycle_p90, 1)} for r in sm.itertuples()}
pr = pd.read_csv(os.path.join(RES, "dryer", "endtime_prediction.csv"))
for ts, g in pr.groupby("test_set"):
    d = g.naive_error_min.abs().values - g.error_min.abs().values
    out["dryer"][f"forecast_{ts}"] = {
        "mae_twin": round(float(g.error_min.abs().mean()), 1),
        "mae_naive": round(float(g.naive_error_min.abs().mean()), 1),
        "mae_reduction_min": [round(v, 1) for v in boot_ci(d, seed=3)],
        "horizon_median_min": round(float(g.horizon_min.median()), 1), "n": int(len(g))}

# sweep and optimisation
out["sweep"] = json.load(open(os.path.join(RES, "sweep", "timing.json")))
s = json.load(open(os.path.join(RES, "optimisation", "summary.json")))
out["optimisation"] = {k: s[k] for k in ("brute_force_best", "brute_force_best_margin", "baseline_margin",
                                          "top10_margin_spread_eur", "surrogate", "surrogate_regret_max",
                                          "random_regret_max", "surrogate_found_true_best",
                                          "random_found_true_best", "budget_runs")}
out["optimisation"]["budget_share_pct"] = 100 * s["budget_runs"] / s["total_runs"]
out["optimisation"]["verification"] = s["verification"]
cb = pd.read_csv(os.path.join(RES, "optimisation", "constrained_best.csv"), dtype={"shifts": str})
out["optimisation"]["constrained_best"] = cb.to_dict(orient="records")
me = pd.read_csv(os.path.join(RES, "optimisation", "main_effects.csv"))
out["optimisation"]["main_effect_range_eur"] = me.groupby("lever").margin_mean.agg(lambda x: x.max() - x.min()).to_dict()

# sequencing
b = pd.read_csv(os.path.join(RES, "sequencing", "benchmark.csv"))
sc = pd.read_csv(os.path.join(RES, "sequencing", "milp_scaling.csv"))
chk = b.dropna(subset=["brute_force_cost"])
out["sequencing"] = {
    "instances_per_size": int(b.groupby("n").size().iloc[0]),
    "milp_equals_enumeration": f"{int(np.isclose(chk.milp_cost, chk.brute_force_cost).sum())}/{len(chk)}",
    "sa_gap_median_pct": b.groupby("n").sa_gap_pct.median().round(1).to_dict(),
    "sa_optimal_count": b.groupby("n").sa_optimal.sum().astype(int).to_dict(),
    "sa_feasible_median_pct": (100 * b.groupby("n").sa_feasible_frac.median()).round(0).to_dict(),
    "milp_median_s_small": round(float(b.milp_s.median()), 2), "milp_max_s_small": round(float(b.milp_s.max()), 2),
    "milp_scaling_median_s": sc.groupby("n").milp_s.median().round(1).to_dict(),
    "milp_scaling_timeouts": sc[sc.status != "Optimal"].groupby("n").size().to_dict(),
}
with open(os.path.join(RES, "report_numbers.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
