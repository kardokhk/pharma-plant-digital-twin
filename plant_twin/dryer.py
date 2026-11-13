"""Level 2 unit-operation twin: fluid-bed dryer with an extended Kalman filter soft sensor.

The "true" process and the twin's internal model deliberately differ (falling-rate exponent,
critical moisture, sensor gain error), so the estimator is tested under model mismatch.

Units: moisture X in kg water per kg dry solid (dry basis); time in minutes; temperature in degC.
LOD (loss on drying, wet basis) = X / (1 + X).
"""
from dataclasses import dataclass

import numpy as np

DT = 1.0 / 6.0          # sample interval, min (10 s)
T_MAX = 150.0           # hard stop, min
X_E = 0.010             # equilibrium moisture
TAU_P = 3.0             # product temperature lag, min
LOD_LOW, LOD_TARGET, LOD_HIGH = 0.010, 0.020, 0.030   # specification band (wet basis)


def lod(x):
    return x / (1.0 + x)


def x_from_lod(w):
    return w / (1.0 - w)


@dataclass
class Batch:
    x0: float        # initial moisture
    k: float         # constant-rate drying rate, 1/min
    xc: float        # critical moisture
    nf: float        # falling-rate exponent
    t_wb: float      # wet-bulb temperature of inlet air
    t_in: float      # inlet air temperature
    gain: float      # exhaust-humidity sensor gain error (multiplicative)
    t_wb_meas: float # measured wet-bulb temperature (from inlet humidity sensor)
    x0_known: float  # initial moisture as known from granulation liquid addition (sd 0.015)


def sample_batch(rng, t_wb_mean=27.0):
    """Draw one batch. t_wb_mean = 27 degC is the calibration climate; 32 degC is a humid summer."""
    t_wb = rng.normal(t_wb_mean, 1.5)
    b = Batch(
        x0=rng.normal(0.25, 0.03),
        k=float(np.exp(rng.normal(np.log(0.0060), 0.20))),
        xc=rng.normal(0.080, 0.010),
        nf=rng.uniform(1.0, 1.6),
        t_wb=t_wb,
        t_in=60.0,
        gain=1.0 + rng.normal(0.0, 0.05),
        t_wb_meas=t_wb + rng.normal(0.0, 0.5),
        x0_known=0.0,
    )
    b.x0_known = b.x0 + rng.normal(0.0, 0.015)
    return b


def _g_true(x, b):
    if x >= b.xc:
        return 1.0
    return max((x - X_E) / (b.xc - X_E), 0.0) ** b.nf


def simulate_truth(b, rng, t_max=T_MAX):
    """Return time grid, true moisture, measured product temperature, measured drying rate."""
    n = int(round(t_max / DT)) + 1
    t = np.arange(n) * DT
    x = np.empty(n)
    tp = np.empty(n)
    x[0], tp[0] = b.x0, b.t_wb + 2.0
    for i in range(1, n):
        g = _g_true(x[i - 1], b)
        x[i] = max(x[i - 1] - b.k * g * DT, X_E)
        t_eq = b.t_wb + (b.t_in - b.t_wb) * (1.0 - g)
        tp[i] = tp[i - 1] + (t_eq - tp[i - 1]) * DT / TAU_P
    rate = np.array([b.k * _g_true(xi, b) for xi in x])
    y_tp = tp + rng.normal(0.0, 0.3, n)
    y_rate = b.gain * rate + rng.normal(0.0, 0.0003, n)
    return t, x, y_tp, y_rate


class DryerEKF:
    """EKF on z = [X, ln k, Tp, Xc] with the twin's nominal falling-rate exponent.

    The initial moisture prior comes from the granulation step (known liquid addition),
    so its spread is smaller than the batch-to-batch spread of the true process.
    """

    NF = 1.3

    def __init__(self, t_wb, t_in=60.0, x0_prior=0.25, x0_sd=0.015):
        self.t_wb, self.t_in = t_wb, t_in
        self.z = np.array([x0_prior, np.log(0.0060), t_wb + 2.0, 0.080])
        self.P = np.diag([x0_sd ** 2, 0.25 ** 2, 1.0 ** 2, 0.012 ** 2])
        self.Q = np.diag([1e-3 ** 2, 0.002 ** 2, 0.05 ** 2, 1e-5 ** 2])
        self.R = np.diag([0.3 ** 2, 0.0005 ** 2])

    def _g(self, x, xc):
        if x >= xc:
            return 1.0
        return max((x - X_E) / (xc - X_E), 0.0) ** self.NF

    def f(self, z):
        x, lk, tp, xc = z
        g = self._g(x, xc)
        k = np.exp(lk)
        t_eq = self.t_wb + (self.t_in - self.t_wb) * (1.0 - g)
        return np.array([max(x - k * g * DT, X_E), lk, tp + (t_eq - tp) * DT / TAU_P, xc])

    def h(self, z):
        x, lk, tp, xc = z
        return np.array([tp, np.exp(lk) * self._g(x, xc)])

    @staticmethod
    def _jac(fun, z, eps=(1e-5, 1e-4, 1e-3, 1e-5)):
        f0 = fun(z)
        J = np.empty((f0.size, z.size))
        for j in range(z.size):
            dz = np.zeros_like(z)
            dz[j] = eps[j]
            J[:, j] = (fun(z + dz) - fun(z - dz)) / (2 * eps[j])
        return J

    def step(self, y):
        F = self._jac(self.f, self.z)
        self.z = self.f(self.z)
        self.P = F @ self.P @ F.T + self.Q
        H = self._jac(self.h, self.z)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + K @ (y - self.h(self.z))
        self.z[0] = max(self.z[0], X_E)
        I_KH = np.eye(self.z.size) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.z[0], np.sqrt(max(self.P[0, 0], 0.0))


def run_estimator(b, y_tp, y_rate):
    ekf = DryerEKF(t_wb=b.t_wb_meas, t_in=b.t_in, x0_prior=b.x0_known)
    xs, sds = [ekf.z[0]], [np.sqrt(ekf.P[0, 0])]
    for i in range(1, len(y_tp)):
        xh, sd = ekf.step(np.array([y_tp[i], y_rate[i]]))
        xs.append(xh)
        sds.append(sd)
    return np.array(xs), np.array(sds)


# ---------------------------------------------------------------------------------------------
# End-point policies. Each returns the stop index given the measured signals (and estimator).
# ---------------------------------------------------------------------------------------------

def stop_fixed_time(t, t_fix):
    return int(min(np.searchsorted(t, t_fix), len(t) - 1))


def stop_product_temp(y_tp, t_stop, window=6):
    """Stop when the moving average of product temperature first reaches t_stop."""
    ma = np.convolve(y_tp, np.ones(window) / window, mode="full")[: len(y_tp)]
    idx = np.nonzero((ma >= t_stop) & (np.arange(len(y_tp)) >= window))[0]
    return int(idx[0]) if idx.size else len(y_tp) - 1


def stop_twin(x_hat, x_sd, lod_stop, z=1.645):
    """Stop when the upper one-sided 95% bound of estimated LOD falls below lod_stop."""
    upper = lod(x_hat + z * x_sd)
    idx = np.nonzero(upper <= lod_stop)[0]
    return int(idx[0]) if idx.size else len(x_hat) - 1


REDRY_MIN = 10.0        # extra drying when the at-line LOD check fails high
CHECK_MIN = 15.0        # at-line LOD check (sampling and analysis) at every stop


def batch_outcome(t, x_true, stop_idx):
    """Apply the at-line LOD check: under-dried batches are re-dried in 10 min steps."""
    i = stop_idx
    dry_min = t[i]
    n_redry = 0
    step = int(round(REDRY_MIN / DT))
    while lod(x_true[i]) > LOD_HIGH and i + step < len(t):
        i += step
        n_redry += 1
        dry_min = t[i]
    cycle = dry_min + CHECK_MIN * (1 + n_redry)
    final = lod(x_true[i])
    return {"dry_min": dry_min, "cycle_min": cycle, "n_redry": n_redry,
            "lod_first": lod(x_true[stop_idx]), "lod_final": final,
            "over_dried": final < LOD_LOW, "under_first": lod(x_true[stop_idx]) > LOD_HIGH}


def predict_end_time(ekf_state, t_wb, t_in, lod_target=LOD_TARGET):
    """Forward-simulate the estimator's mean model from its current state; minutes to target LOD."""
    ekf = DryerEKF(t_wb=t_wb, t_in=t_in)
    z = np.array(ekf_state, dtype=float)
    x_target = x_from_lod(lod_target)
    for i in range(int(T_MAX / DT)):
        if z[0] <= x_target:
            return i * DT
        z = ekf.f(z)
    return T_MAX


def run_estimator_states(b, y_tp, y_rate):
    """Like run_estimator but also returns the full state trajectory."""
    ekf = DryerEKF(t_wb=b.t_wb_meas, t_in=b.t_in, x0_prior=b.x0_known)
    zs, sds = [ekf.z.copy()], [np.sqrt(ekf.P[0, 0])]
    for i in range(1, len(y_tp)):
        ekf.step(np.array([y_tp[i], y_rate[i]]))
        zs.append(ekf.z.copy())
        sds.append(np.sqrt(max(ekf.P[0, 0], 0.0)))
    return np.array(zs), np.array(sds)
