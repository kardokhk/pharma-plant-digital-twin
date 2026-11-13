"""Small statistics helpers used across the experiments."""
import numpy as np


def wilson(k, n, z=1.959964):
    """Wilson score 95% interval for a proportion k/n, returned as percentages."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def boot_ci(x, stat=np.mean, n_boot=4000, seed=0, alpha=0.05):
    """Percentile bootstrap: returns (estimate, lower, upper)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    reps = np.apply_along_axis(stat, 1, x[idx])
    return (float(stat(x)), float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2)))


def paired_diff_ci(a, b, n_boot=4000, seed=0):
    """Mean of a - b over paired replications with a bootstrap 95% CI."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    return boot_ci(diff, np.mean, n_boot=n_boot, seed=seed)
