"""Level 6: campaign sequencing with sequence-dependent cleaning, three ways.

Problem: order n products on a shared line, starting from the product currently on the line,
to minimise total changeover (cleaning plus set-up) hours. This is an open asymmetric travelling
salesman path.

  exact_milp   : MILP with Miller-Tucker-Zemlin subtour elimination, solved by HiGHS
  brute_force  : enumeration, used only to check the MILP on small instances
  to_qubo      : one-hot position encoding, n*n binary variables, the input form accepted by
                 quantum annealers and QAOA
  anneal_qubo  : classical simulated annealing on that QUBO (a stand-in; no quantum hardware used)
"""
import itertools
import time

import highspy
import numpy as np


def make_instance(n, rng):
    """Changeover matrix (hours) from product attributes. Index 0 is the product now on the line."""
    m = n + 1
    potency = rng.random(m) < 0.2                 # high-potency products need full decontamination
    colour = rng.integers(0, 4, m)                # coating shade 0 (light) .. 3 (dark)
    allergen = rng.random(m) < 0.25
    tooling = rng.integers(0, 3, m)               # tablet press tooling set
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            h = 2.0
            h += 1.5 * max(colour[i] - colour[j], 0) + 0.3 * max(colour[j] - colour[i], 0)
            h += 3.0 if tooling[i] != tooling[j] else 0.0
            h += 2.5 if (allergen[i] and not allergen[j]) else 0.0
            h += 6.0 if potency[i] else 0.0
            h += rng.uniform(0.0, 0.5)
            C[i, j] = round(h, 2)
    return C


def path_cost(C, order):
    """order: sequence of product indices 1..n; the path starts at 0."""
    prev, tot = 0, 0.0
    for j in order:
        tot += C[prev, j]
        prev = j
    return tot


def brute_force(C):
    n = C.shape[0] - 1
    best, arg = np.inf, None
    for perm in itertools.permutations(range(1, n + 1)):
        c = path_cost(C, perm)
        if c < best:
            best, arg = c, perm
    return best, list(arg)


def exact_milp(C, time_limit=120.0):
    """Open ATSP path from node 0 via MTZ. Returns (cost, order, seconds, status)."""
    m = C.shape[0]
    n = m - 1
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("random_seed", 0)
    inf = highspy.kHighsInf
    # variables x_ij for i != j, j != 0 (no return to start), plus u_i for i = 1..n
    arcs = [(i, j) for i in range(m) for j in range(1, m) if i != j]
    idx = {a: k for k, a in enumerate(arcs)}
    for (i, j) in arcs:
        h.addVar(0.0, 1.0)
    nx = len(arcs)
    for _ in range(n):
        h.addVar(1.0, float(n))
    h.changeColsCost(nx, np.arange(nx, dtype=np.int32), np.array([C[i, j] for i, j in arcs]))
    h.changeColsIntegrality(nx, np.arange(nx, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * nx))
    # each product entered exactly once
    for j in range(1, m):
        cols = [idx[(i, j)] for i in range(m) if i != j]
        h.addRow(1.0, 1.0, len(cols), np.array(cols, dtype=np.int32), np.ones(len(cols)))
    # start node leaves once; each product leaves at most once (the last one leaves zero times)
    cols = [idx[(0, j)] for j in range(1, m)]
    h.addRow(1.0, 1.0, len(cols), np.array(cols, dtype=np.int32), np.ones(len(cols)))
    for i in range(1, m):
        cols = [idx[(i, j)] for j in range(1, m) if j != i]
        h.addRow(0.0, 1.0, len(cols), np.array(cols, dtype=np.int32), np.ones(len(cols)))
    # MTZ: u_i - u_j + n x_ij <= n - 1 for i, j in 1..n
    for (i, j) in arcs:
        if i == 0:
            continue
        ui, uj = nx + i - 1, nx + j - 1
        h.addRow(-inf, float(n - 1), 3, np.array([ui, uj, idx[(i, j)]], dtype=np.int32),
                 np.array([1.0, -1.0, float(n)]))
    t0 = time.time()
    h.run()
    secs = time.time() - t0
    status = h.modelStatusToString(h.getModelStatus())
    sol = np.array(h.getSolution().col_value)
    succ = {i: j for (i, j), k in idx.items() if sol[k] > 0.5}
    order, cur = [], 0
    while cur in succ and len(order) < n:
        cur = succ[cur]
        order.append(cur)
    return path_cost(C, order), order, secs, status


def to_qubo(C, penalty=None):
    """QUBO over x[i, t] (product i+1 at position t). Returns symmetric Q and constant offset."""
    m = C.shape[0]
    n = m - 1
    N = n * n
    A = penalty if penalty is not None else 1.5 * C.max() * 2

    def v(i, t):
        return i * n + t

    Q = np.zeros((N, N))
    const = 0.0
    # one position per product and one product per position: A (1 - sum)^2
    for i in range(n):
        group = [v(i, t) for t in range(n)]
        const += A
        for a in group:
            Q[a, a] -= A
        for a, b in itertools.combinations(group, 2):
            Q[a, b] += A
            Q[b, a] += A
    for t in range(n):
        group = [v(i, t) for i in range(n)]
        const += A
        for a in group:
            Q[a, a] -= A
        for a, b in itertools.combinations(group, 2):
            Q[a, b] += A
            Q[b, a] += A
    # changeover from the current product to position 0
    for i in range(n):
        Q[v(i, 0), v(i, 0)] += C[0, i + 1]
    # changeovers between consecutive positions
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    a, b = v(i, t), v(j, t + 1)
                    Q[a, b] += C[i + 1, j + 1] / 2
                    Q[b, a] += C[i + 1, j + 1] / 2
    return Q, const


def decode(x, n):
    X = x.reshape(n, n)
    feasible = bool((X.sum(0) == 1).all() and (X.sum(1) == 1).all())
    order = [int(np.argmax(X[:, t])) + 1 for t in range(n)] if feasible else None
    return feasible, order


def anneal_qubo(Q, n_chains=256, sweeps=3000, beta0=0.01, beta1=5.0, seed=0):
    """Single-flip Metropolis simulated annealing, many chains in parallel (vectorised).

    Energy E(x) = x^T Q x with symmetric Q. Returns the final states of all chains.
    """
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    diag = np.diag(Q).copy()
    Qoff = Q - np.diag(diag)
    x = (rng.random((n_chains, N)) < 0.5).astype(float)
    field = x @ Qoff                     # sum_j Q_kj x_j for each chain
    scale = np.abs(Q).max()
    betas = np.geomspace(beta0, beta1, sweeps) / scale
    for beta in betas:
        for k in rng.permutation(N):
            xk = x[:, k]
            dE = (1 - 2 * xk) * (diag[k] + 2 * field[:, k])
            accept = (dE <= 0) | (rng.random(n_chains) < np.exp(-beta * np.clip(dE, 0, None)))
            if accept.any():
                delta = np.where(accept, 1 - 2 * xk, 0.0)
                x[:, k] = xk + delta
                field += np.outer(delta, Qoff[k])
    return x
