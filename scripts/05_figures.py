"""Figures for the plant-twin demo report. Reads results/, writes figures/ (PDF + 600 dpi PNG)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_twin import plotstyle as figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
FAM = figstyle.use_print_style()
OI = figstyle.OKABE_ITO
WIDTH_MM = 160  # final width in the A4 report (2.54 cm margins); never rescaled afterwards
POL = {"fixed": ("Fixed time", OI[1], "s"), "temp": ("Product temperature", OI[0], "o"),
       "twin": ("Twin estimate", OI[2], "^")}


def fig_mm(w, h):
    assert h <= figstyle.MAX_HEIGHT_MM
    return plt.figure(figsize=(w / 25.4, h / 25.4), layout="constrained")


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=600)
    plt.close(fig)


def eur_m(x, _=None):
    return f"{x / 1e6:.1f}"


# ------------------------------------------------------------------------------------------
# Figure 1: Level 2 dryer twin
# ------------------------------------------------------------------------------------------
def fig_dryer():
    tr = pd.read_csv(os.path.join(RES, "dryer", "example_trace.csv"))
    sm = pd.read_csv(os.path.join(RES, "dryer", "summary.csv"))
    tb = pd.read_csv(os.path.join(RES, "dryer", "test_batches.csv"))
    pr = pd.read_csv(os.path.join(RES, "dryer", "endtime_prediction.csv"))
    fig = fig_mm(WIDTH_MM, 64)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.8])
    ax = fig.add_subplot(gs[0])
    m = tr.t_min <= 90
    ax.axhspan(1.0, 3.0, color="0.92", lw=0)
    ax.text(88, 4.2, "Specification 1.0-3.0%", ha="right", va="bottom", fontsize=6, color="0.35")
    ax.fill_between(tr.t_min[m], 100 * tr.lod_lower95[m], 100 * tr.lod_upper95[m], color=OI[2], alpha=0.25, lw=0,
                    label="Twin 90% band")
    ax.plot(tr.t_min[m], 100 * tr.lod_true[m], color="k", lw=1.0, label="True moisture")
    ax.plot(tr.t_min[m], 100 * tr.lod_est[m], color=OI[2], lw=1.0, ls="--", label="Twin estimate")
    ax.set_xlabel("Drying time (min)")
    ax.set_ylabel("Moisture, LOD (%)")
    ax.set_ylim(0, 22)
    ax.set_xlim(0, 90)
    ax.legend(frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[1])
    st = tb[tb.test_set == "stable"]
    for i, p in enumerate(("fixed", "temp", "twin")):
        v = st[st.policy == p].cycle_min.values
        parts = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(POL[p][1])
            b.set_alpha(0.45)
            b.set_edgecolor("none")
        q = np.percentile(v, [10, 50, 90])
        ax.plot([i, i], [q[0], q[2]], color="k", lw=0.8)
        ax.plot(i, q[1], marker=POL[p][2], color="k", ms=3.5)
        row = sm[(sm.test_set == "stable") & (sm.policy == p)].iloc[0]
        ax.text(i, 140, f"{row.in_spec_pct:.1f}%\nin spec", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(3), [POL[p][0].replace(" ", "\n", 1) for p in ("fixed", "temp", "twin")])
    ax.set_ylabel("Cycle time incl. LOD checks (min)")
    ax.set_ylim(40, 160)

    ax = fig.add_subplot(gs[2])
    st = pr[pr.test_set == "stable"]
    data = [st.naive_error_min.abs().values, st.error_min.abs().values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True,
                    medianprops={"color": "k", "lw": 1.0})
    for b, c in zip(bp["boxes"], ("0.75", OI[2])):
        b.set_facecolor(c)
        b.set_alpha(0.6)
    ax.set_xticks([0, 1], ["Historical\nmedian", "Twin\nforecast"])
    ax.set_ylabel("Absolute error of end-time forecast (min)")
    ax.set_ylim(0, 45)
    figstyle.panel_labels(fig.axes)
    save(fig, "fig1_dryer_twin")


# ------------------------------------------------------------------------------------------
# Figure 2: Level 5/6 plant twin and optimisation
# ------------------------------------------------------------------------------------------
def fig_plant():
    c = pd.read_csv(os.path.join(RES, "optimisation", "config_means.csv"), dtype={"shifts": str})
    me = pd.read_csv(os.path.join(RES, "optimisation", "main_effects.csv"))
    sr = pd.read_csv(os.path.join(RES, "optimisation", "surrogate_vs_random.csv"))
    sm = json.load(open(os.path.join(RES, "optimisation", "summary.json")))
    fig = fig_mm(WIDTH_MM, 128)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

    levers = [("shifts", "Shifts", ["2", "3", "24/7"]), ("campaign", "Campaign\nlength", None),
              ("conwip", "WIP cap", None), ("qc_analysts", "QC\nanalysts", None),
              ("pm_interval", "Press PM\ninterval (h)", None), ("dryer", "Dryer\nend-point", ["fixed", "temp", "twin"]),
              ("coaters", "Coaters", None)]
    sizes = [5.5, 8, 10, 4.5, 6, 5.5, 3.5]
    sub = gs[0, :2].subgridspec(1, len(levers), width_ratios=[max(n, 3) for n in sizes], wspace=0.05)
    me_axes = []
    for k, (lever, name, order) in enumerate(levers):
        ax = fig.add_subplot(sub[k], sharey=me_axes[0] if me_axes else None)
        me_axes.append(ax)
        d = me[me.lever == lever].copy()
        if order:
            d = d.set_index("level").loc[order].reset_index()
        xs = np.arange(len(d))
        ax.plot(xs, d.margin_mean / 1e6, color=OI[0], marker="o", ms=2.5, lw=0.9)
        lab = [str(v).replace(".0", "") for v in d.level]
        if lever == "dryer":
            lab = ["fixed", "temp", "twin"]
        if lever == "pm_interval":
            lab = [l if i in (0, len(lab) - 1) else "" for i, l in enumerate(lab)]
        if lever == "conwip":
            lab = [l if i % 3 == 0 else "" for i, l in enumerate(lab)]
        ax.set_xticks(xs, lab, fontsize=5.5, rotation=40 if lever == "dryer" else 0,
                      ha="right" if lever == "dryer" else "center")
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_title(name, fontsize=6)
        if k:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color="0.9", lw=0.5)
    me_axes[0].set_ylabel("Mean quarterly margin\n(EUR million)")
    me_axes[0].set_ylim(2.2, 4.5)

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(c.cost_mean / 1e6, 100 * c.service_mean, s=1, color="0.6", alpha=0.25, rasterized=True, lw=0)
    pf = pd.read_csv(os.path.join(RES, "optimisation", "pareto_service_cost.csv"))
    ax.plot(pf.cost_mean / 1e6, 100 * pf.service_mean, color=OI[0], lw=1.0, label="Efficient frontier")
    v = sm["verification"]
    for key, lab, col, mk in (("baseline", "Baseline", OI[1], "s"), ("brute_force_best", "Best margin", OI[2], "^")):
        ax.plot(v[key]["cost"][0] / 1e6, 100 * v[key]["service"][0], marker=mk, color=col, ms=5, ls="none",
                mec="k", mew=0.5, label=lab)
    ax.set_xlabel("Operating cost\n(EUR million per quarter)")
    ax.set_ylabel("Service level (% of demand met)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1, fontsize=6, labelspacing=0.3)

    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    for i, (col, lab, colr) in enumerate((("regret_eur", "Surrogate", OI[2]), ("random_regret_eur", "Random", OI[1]))):
        y = sr[col].values / 1e3
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=8, color=colr, alpha=0.8, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color="k", lw=1.0)
    ax.set_xticks([0, 1], ["Surrogate\nloop", "Random\nsearch"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Shortfall from true best (EUR thousand)")
    ax.set_ylim(bottom=-5)

    ax = fig.add_subplot(gs[1, 1:])
    runs = np.load(os.path.join(RES, "optimisation", "verification_runs.npy"))
    names = ["Baseline", "Best margin"]
    cb = pd.read_csv(os.path.join(RES, "optimisation", "constrained_best.csv"), dtype={"shifts": str})
    for i, nm in enumerate(names):
        y = runs[[0, 1][i], :, 0] / 1e6
        parts = ax.violinplot(y, positions=[i], widths=0.7, showextrema=False, orientation="vertical")
        for b in parts["bodies"]:
            b.set_facecolor([OI[1], OI[2]][i])
            b.set_alpha(0.5)
            b.set_edgecolor("none")
        ax.plot(i, np.median(y), marker="o", color="k", ms=3)
    row = cb[(cb.shifts == "3") & (cb.coaters == 1)].iloc[0]
    ax.axhline(row.margin_mean / 1e6, color=OI[0], lw=0.8, ls="--")
    ax.text(1.45, row.margin_mean / 1e6, "Best with no new\nequipment (30 reps)", fontsize=6, color=OI[0],
            va="center", ha="left")
    ax.set_xticks([0, 1], ["Baseline\n(campaign 2,\nWIP cap 6)", "Best margin\n(campaign 6, WIP cap 11,\n2nd coater)"])
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylabel("Quarterly margin, 200 fresh replications\n(EUR million)")
    axes = [me_axes[0]] + [a for a in fig.axes if a not in me_axes]
    figstyle.panel_labels(axes[:1], ["A"], dx=-0.9, dy=1.12)
    figstyle.panel_labels(axes[1:], ["B", "C", "D"])
    save(fig, "fig2_plant_optimisation")


# ------------------------------------------------------------------------------------------
# Figure 3: sequencing, MILP vs QUBO
# ------------------------------------------------------------------------------------------
def fig_seq():
    b = pd.read_csv(os.path.join(RES, "sequencing", "benchmark.csv"))
    s = pd.read_csv(os.path.join(RES, "sequencing", "milp_scaling.csv"))
    fig = fig_mm(WIDTH_MM, 58)
    gs = fig.add_gridspec(1, 3)
    rng = np.random.default_rng(1)
    ax = fig.add_subplot(gs[0])
    for n, g in b.groupby("n"):
        ax.scatter(n + rng.uniform(-0.2, 0.2, len(g)), g.sa_gap_pct, s=5, color=OI[1], alpha=0.7, lw=0)
    med = b.groupby("n").sa_gap_pct.median()
    ax.plot(med.index, med.values, color="k", lw=1.0, marker="o", ms=2.5, label="Median")
    ax.axhline(0, color=OI[0], lw=0.8, ls="--", label="Exact MILP optimum")
    ax.set_xlabel("Products to sequence")
    ax.set_ylabel("QUBO annealing gap to optimum (%)")
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1])
    f = b.groupby("n").sa_feasible_frac.median()
    ax.plot(f.index, 100 * f.values, color=OI[1], marker="s", ms=2.5)
    ax2 = ax
    ax2.set_xlabel("Products to sequence")
    ax2.set_ylabel("Valid annealing results (%)")
    ax2.set_ylim(0, 100)
    for n in (6, 9, 12):
        ax2.text(n, 100 * f.loc[n] + 5, f"{n * n} binary\nvariables", ha="center", fontsize=5.5)

    ax = fig.add_subplot(gs[2])
    milp_all = pd.concat([b[["n", "milp_s"]], s[["n", "milp_s"]]])
    for n, g in milp_all.groupby("n"):
        ax.scatter(n + rng.uniform(-0.3, 0.3, len(g)), g.milp_s, s=5, color=OI[0], alpha=0.7, lw=0)
    ax.set_yscale("log")
    ax.axhline(300, color="0.5", lw=0.6, ls=":")
    ax.text(5, 330, "300 s limit", fontsize=6, color="0.4")
    ax.set_xlabel("Products to sequence")
    ax.set_ylabel("Exact MILP solve time (s)")
    figstyle.panel_labels(fig.axes)
    save(fig, "fig3_sequencing")


if __name__ == "__main__":
    print("font:", FAM)
    fig_dryer()
    fig_plant()
    fig_seq()
