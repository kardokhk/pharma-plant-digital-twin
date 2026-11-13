"""Level 5/6: brute-force scenario sweep of the plant twin on one HPC node.

Every combination of seven operating levers (34,560 configurations) is simulated for one quarter
(13 weeks after a 2-week warm-up) with R replications. Replication r uses seed r in every
configuration (common random numbers), so configurations are compared on paired noise.

Output: results/sweep/runs.npz (per-run KPIs), results/sweep/configs.csv, results/sweep/timing.json
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin.plant import Config, simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "sweep")
R = int(os.environ.get("REPS", 30))

LEVERS = {
    "shifts": [2, 3, "24/7"],
    "qc_analysts": [2, 3, 4, 5],
    "campaign": [1, 2, 3, 4, 5, 6, 7, 8],
    "conwip": list(range(3, 13)),
    "pm_interval": [100.0, 140.0, 180.0, 220.0, 260.0, 320.0],
    "dryer": ["fixed", "temp", "twin"],
    "coaters": [1, 2],
}
KPIS = ["margin_eur", "service_level", "released", "lead_time_h_mean", "cost_eur",
        "equipment_changeovers", "breakdowns", "util_PACK", "util_COMP", "util_COAT"]

_DRYER = None


def _init(dryer_samples):
    global _DRYER
    _DRYER = dryer_samples


def run_config(args):
    idx, cfg_dict = args
    cfg = Config(**cfg_dict)
    out = np.empty((R, len(KPIS)))
    for r in range(R):
        k = simulate(cfg, r, _DRYER)
        out[r] = [k[name] for name in KPIS]
    return idx, out


def all_configs():
    names = list(LEVERS)
    return [dict(zip(names, combo)) for combo in itertools.product(*LEVERS.values())]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "results", "dryer", "cycle_samples.json")) as fh:
        dryer = json.load(fh)
    cfgs = all_configs()
    nproc = int(os.environ.get("NPROC", os.cpu_count()))
    t0 = time.time()
    res = np.empty((len(cfgs), R, len(KPIS)))
    with Pool(nproc, initializer=_init, initargs=(dryer,)) as pool:
        for i, (idx, out) in enumerate(pool.imap_unordered(run_config, list(enumerate(cfgs)), chunksize=8)):
            res[idx] = out
            if i % 2000 == 0:
                print(f"{i}/{len(cfgs)} configs, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    np.savez_compressed(os.path.join(OUT, "runs.npz"), res=res, kpis=np.array(KPIS))
    pd.DataFrame(cfgs).to_csv(os.path.join(OUT, "configs.csv"), index_label="config_id")
    with open(os.path.join(OUT, "timing.json"), "w") as fh:
        json.dump({"configs": len(cfgs), "reps": R, "runs": len(cfgs) * R, "processes": nproc,
                   "wall_s": wall, "host": os.uname().nodename}, fh, indent=1)
    print(f"done: {len(cfgs) * R} runs in {wall:.0f} s on {nproc} processes")


if __name__ == "__main__":
    main()
