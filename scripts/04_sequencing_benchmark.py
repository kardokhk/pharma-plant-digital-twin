"""Level 6 benchmark: exact MILP vs QUBO + simulated annealing for campaign sequencing.

For n = 5..12 products, 20 random instances each (seeded). The MILP is checked against
enumeration for n <= 8. Exact MILP scaling is also timed for n = 15, 20, 25, 30.
Output: results/sequencing/benchmark.csv, results/sequencing/milp_scaling.csv
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin import sequencing as s  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "sequencing")
SIZES = range(5, 13)
N_INST = 20
CHAINS, SWEEPS = 256, 2000


def one(args):
    n, k = args
    rng = np.random.default_rng([n, k])
    C = s.make_instance(n, rng)
    milp_cost, order, secs, status = s.exact_milp(C)
    bf = s.brute_force(C)[0] if n <= 8 else np.nan
    Q, const = s.to_qubo(C)
    import time
    t0 = time.time()
    X = s.anneal_qubo(Q, n_chains=CHAINS, sweeps=SWEEPS, seed=k)
    sa_secs = time.time() - t0
    feas = [s.decode(x, n) for x in X]
    costs = [s.path_cost(C, o) for f, o in feas if f]
    best = min(costs) if costs else np.nan
    return {"n": n, "instance": k, "qubo_vars": n * n, "milp_cost": milp_cost, "milp_s": secs,
            "milp_status": status, "brute_force_cost": bf, "sa_best_cost": best,
            "sa_feasible_frac": len(costs) / len(X), "sa_s": sa_secs,
            "sa_gap_pct": 100 * (best - milp_cost) / milp_cost if costs else np.nan,
            "sa_optimal": bool(costs) and abs(best - milp_cost) < 1e-6}


def scaling(args):
    n, k = args
    C = s.make_instance(n, np.random.default_rng([1000 + n, k]))
    cost, order, secs, status = s.exact_milp(C, time_limit=300.0)
    return {"n": n, "instance": k, "milp_s": secs, "status": status}


def main():
    os.makedirs(OUT, exist_ok=True)
    nproc = int(os.environ.get("NPROC", os.cpu_count()))
    with Pool(nproc) as pool:
        rows = pool.map(one, [(n, k) for n in SIZES for k in range(N_INST)], chunksize=1)
        sc = pool.map(scaling, [(n, k) for n in (15, 20, 25, 30) for k in range(10)], chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    pd.DataFrame(sc).to_csv(os.path.join(OUT, "milp_scaling.csv"), index=False)
    chk = df.dropna(subset=["brute_force_cost"])
    print("MILP equals enumeration:", bool(np.allclose(chk.milp_cost, chk.brute_force_cost)), len(chk))
    print(df.groupby("n")[["milp_s", "sa_gap_pct", "sa_optimal", "sa_feasible_frac", "sa_s"]].median())
    print(pd.DataFrame(sc).groupby("n").milp_s.describe())


if __name__ == "__main__":
    main()
