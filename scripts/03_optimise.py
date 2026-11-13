"""Level 6: surrogate-assisted optimisation of the plant twin, checked against brute force.

1. Brute-force optimum of mean quarterly margin over the full sweep (30 replications each).
2. Surrogate loop with a small simulation budget: 1,000 random configurations x 5 replications,
   gradient-boosted surrogate, top 25 predicted configurations re-simulated with 30 replications,
   best of those chosen. Repeated with 20 random seeds. Compared with random search given the
   same number of simulation runs.
3. Verification: baseline, brute-force best and surrogate choice re-simulated with 200 fresh
   replications (seeds 1000-1199, not used anywhere else), paired by seed.
4. Decision tables: best configuration under constraints (shift pattern, second coater) and the
   service-versus-cost frontier.
Output: results/optimisation/*
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin.plant import Config, simulate  # noqa: E402
from plant_twin.stats import boot_ci  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "optimisation")
BASELINE = {"shifts": 3, "qc_analysts": 3, "campaign": 2, "conwip": 6, "pm_interval": 180.0,
            "dryer": "fixed", "coaters": 1}
N_SAMPLE, R_SAMPLE, TOP_K, R_FULL, N_SEEDS = 1000, 5, 25, 30, 20
VERIFY_SEEDS = range(1000, 1200)

_DRYER = None


def _init(d):
    global _DRYER
    _DRYER = d


def _sim(args):
    cfg, seed = args
    k = simulate(Config(**cfg), seed, _DRYER)
    return k["margin_eur"], k["service_level"], k["lead_time_h_mean"], k["cost_eur"], k["released"]


def features(cfgs):
    X = pd.DataFrame({
        "shifts_hours": cfgs.shifts.map({"2": 80, "3": 120, "24/7": 168}).astype(float),
        "qc_analysts": cfgs.qc_analysts, "campaign": cfgs.campaign, "conwip": cfgs.conwip,
        "pm_interval": cfgs.pm_interval, "coaters": cfgs.coaters,
        "dryer_fixed": (cfgs.dryer == "fixed").astype(float),
        "dryer_temp": (cfgs.dryer == "temp").astype(float),
    })
    return X.values


def to_cfg(row):
    sh = row["shifts"]
    return {"shifts": sh if sh == "24/7" else int(sh), "qc_analysts": int(row["qc_analysts"]),
            "campaign": int(row["campaign"]), "conwip": int(row["conwip"]),
            "pm_interval": float(row["pm_interval"]), "dryer": row["dryer"], "coaters": int(row["coaters"])}


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "results", "sweep", "runs.npz"))
    res, kpis = z["res"], list(z["kpis"])
    cfgs = pd.read_csv(os.path.join(ROOT, "results", "sweep", "configs.csv"), dtype={"shifts": str})
    im, isv, ic, il = (kpis.index(k) for k in ("margin_eur", "service_level", "cost_eur", "lead_time_h_mean"))
    mean = res.mean(axis=1)
    cfgs["margin_mean"] = mean[:, im]
    cfgs["margin_sd"] = res[:, :, im].std(axis=1, ddof=1)
    cfgs["service_mean"] = mean[:, isv]
    cfgs["cost_mean"] = mean[:, ic]
    cfgs["lead_mean"] = mean[:, il]
    cfgs.to_csv(os.path.join(OUT, "config_means.csv"), index=False)

    true_best = int(cfgs.margin_mean.idxmax())
    base_mask = np.ones(len(cfgs), bool)
    for k, v in BASELINE.items():
        base_mask &= cfgs[k].astype(str).values == str(v)
    base_id = int(np.nonzero(base_mask)[0][0])

    # ---- surrogate loop vs random search -------------------------------------------------
    X = features(cfgs)
    rows = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        pick = rng.choice(len(cfgs), N_SAMPLE, replace=False)
        y = res[pick, :R_SAMPLE, im].mean(axis=1)
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=s)
        model.fit(X[pick], y)
        pred = model.predict(X)
        unseen = np.setdiff1d(np.arange(len(cfgs)), pick)
        resid = pred[unseen] - cfgs.margin_mean.values[unseen]
        r2 = 1 - (resid ** 2).sum() / ((cfgs.margin_mean.values[unseen] - cfgs.margin_mean.values[unseen].mean()) ** 2).sum()
        top = np.argsort(-pred)[:TOP_K]
        chosen = int(top[np.argmax(cfgs.margin_mean.values[top])])
        budget = N_SAMPLE * R_SAMPLE + TOP_K * R_FULL
        rs = rng.choice(len(cfgs), budget // R_FULL, replace=False)
        rs_best = int(rs[np.argmax(cfgs.margin_mean.values[rs])])
        rank = int((cfgs.margin_mean.values > cfgs.margin_mean.values[chosen]).sum()) + 1
        rows.append({"seed": s, "budget_runs": budget, "surrogate_r2_unseen": r2,
                     "surrogate_mae_eur": float(np.abs(resid).mean()), "chosen": chosen, "chosen_rank": rank,
                     "regret_eur": cfgs.margin_mean[true_best] - cfgs.margin_mean[chosen],
                     "random_best": rs_best,
                     "random_regret_eur": cfgs.margin_mean[true_best] - cfgs.margin_mean[rs_best]})
    sur = pd.DataFrame(rows)
    sur.to_csv(os.path.join(OUT, "surrogate_vs_random.csv"), index=False)
    sur_choice = int(sur.chosen.mode().iloc[0])

    # ---- verification with fresh seeds ---------------------------------------------------
    with open(os.path.join(ROOT, "results", "dryer", "cycle_samples.json")) as fh:
        dryer = json.load(fh)
    ver_ids = {"baseline": base_id, "brute_force_best": true_best, "surrogate_choice": sur_choice}
    jobs = [(to_cfg(cfgs.loc[i]), sd) for i in ver_ids.values() for sd in VERIFY_SEEDS]
    with Pool(int(os.environ.get("NPROC", os.cpu_count())), initializer=_init, initargs=(dryer,)) as pool:
        out = np.array(pool.map(_sim, jobs, chunksize=4)).reshape(len(ver_ids), len(VERIFY_SEEDS), 5)
    ver = {}
    for j, (name, cid) in enumerate(ver_ids.items()):
        ver[name] = {"config": to_cfg(cfgs.loc[cid]), "config_id": cid,
                     "margin": boot_ci(out[j, :, 0], seed=11), "service": boot_ci(out[j, :, 1], seed=11),
                     "lead_h": boot_ci(out[j, :, 2], seed=11), "cost": boot_ci(out[j, :, 3], seed=11),
                     "released": boot_ci(out[j, :, 4], seed=11)}
    for name in ("brute_force_best", "surrogate_choice"):
        j = list(ver_ids).index(name)
        ver[name]["margin_gain_vs_baseline"] = boot_ci(out[j, :, 0] - out[0, :, 0], seed=12)
        ver[name]["service_gain_vs_baseline"] = boot_ci(out[j, :, 1] - out[0, :, 1], seed=12)
    np.save(os.path.join(OUT, "verification_runs.npy"), out)

    # ---- decision tables -------------------------------------------------------------------
    cons = []
    for sh in ("2", "3", "24/7"):
        for co in (1, 2):
            sub = cfgs[(cfgs.shifts == sh) & (cfgs.coaters == co)]
            b = sub.loc[sub.margin_mean.idxmax()]
            cons.append({"shifts": sh, "coaters": co, **{k: b[k] for k in ("qc_analysts", "campaign", "conwip",
                         "pm_interval", "dryer")}, "margin_mean": b.margin_mean, "service_mean": b.service_mean,
                         "cost_mean": b.cost_mean, "lead_mean": b.lead_mean})
    pd.DataFrame(cons).to_csv(os.path.join(OUT, "constrained_best.csv"), index=False)

    order = cfgs.sort_values(["cost_mean", "service_mean"], ascending=[True, False])
    front, best_s = [], -1.0
    for i, r in order.iterrows():
        if r.service_mean > best_s + 1e-9:
            front.append(i)
            best_s = r.service_mean
    cfgs.loc[front].to_csv(os.path.join(OUT, "pareto_service_cost.csv"))

    # main effects (mean margin by lever level, averaged over all other levers)
    eff = []
    for lever in ("shifts", "qc_analysts", "campaign", "conwip", "pm_interval", "dryer", "coaters"):
        g = cfgs.groupby(lever).margin_mean.mean()
        for lvl, v in g.items():
            eff.append({"lever": lever, "level": str(lvl), "margin_mean": v})
    pd.DataFrame(eff).to_csv(os.path.join(OUT, "main_effects.csv"), index=False)

    summary = {
        "n_configs": len(cfgs), "reps": int(res.shape[1]), "total_runs": int(res.shape[0] * res.shape[1]),
        "brute_force_best": to_cfg(cfgs.loc[true_best]), "brute_force_best_margin": cfgs.margin_mean[true_best],
        "baseline_margin": cfgs.margin_mean[base_id],
        "top10_margin_spread_eur": float(np.ptp(np.sort(cfgs.margin_mean.values)[-10:])),
        "surrogate": {k: float(v) for k, v in sur[["surrogate_r2_unseen", "surrogate_mae_eur", "regret_eur",
                                                   "random_regret_eur", "chosen_rank"]].median().items()},
        "surrogate_regret_max": float(sur.regret_eur.max()), "random_regret_max": float(sur.random_regret_eur.max()),
        "surrogate_found_true_best": int((sur.chosen == true_best).sum()),
        "random_found_true_best": int((sur.random_best == true_best).sum()),
        "budget_runs": int(sur.budget_runs.iloc[0]),
        "verification": ver,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
