"""Level 5 plant twin: discrete-event simulation of a hypothetical oral solid dose (OSD) site.

Route per batch: dispensing -> granulation -> fluid-bed drying -> blending -> compression ->
film coating (coated products only) -> packaging -> QC testing and release.

Modelled: sequence-dependent cleaning, random breakdowns, age-based tablet-press wear with
preventive maintenance, shift calendars, a QC laboratory with day-shift analysts, CONWIP release
control and campaign sequencing. All parameter values are illustrative, not from a real plant.
Time unit: hours. Simulation starts on a Monday at 06:00.
"""
from dataclasses import dataclass, asdict, field
import json

import numpy as np
import simpy

WEEK = 168.0

# --------------------------------------------------------------------------------------------
# Plant definition
# --------------------------------------------------------------------------------------------
PRODUCTS = ["P1", "P2", "P3", "P4", "P5"]
DEMAND_SHARE = {"P1": 0.30, "P2": 0.25, "P3": 0.20, "P4": 0.15, "P5": 0.10}
COATED = {"P1": True, "P2": True, "P3": False, "P4": True, "P5": True}
POTENT = {"P5"}                                       # high-potency product: longer cleaning
COMP_H = {"P1": 9.0, "P2": 7.0, "P3": 6.0, "P4": 8.0, "P5": 7.5}   # compression, median h
CONTRIB_EUR = {"P1": 30e3, "P2": 28e3, "P3": 20e3, "P4": 35e3, "P5": 45e3}  # contribution/batch
DEMAND_Q = {"P1": 60, "P2": 50, "P3": 40, "P4": 30, "P5": 20}  # batches per quarter; excess earns nothing
BATCH_VALUE_EUR = 150e3                              # inventory value of a released batch

# step: (equipment, median hours, CV)
STEPS = [("DISP", 2.0, 0.10), ("GRAN", 2.5, 0.10), ("DRY", None, None), ("BLEND", 1.5, 0.10),
         ("COMP", None, 0.12), ("COAT", 5.0, 0.10), ("PACK", 6.0, 0.10)]
EQUIP_UNITS = {"DISP": 1, "GRAN": 1, "DRY": 1, "BLEND": 1, "COMP": 2, "COAT": 1, "PACK": 1}
MINOR_CLEAN_H = {"DISP": 0.5, "GRAN": 1.0, "DRY": 0.75, "BLEND": 0.75, "COMP": 1.5, "COAT": 1.0, "PACK": 1.0}
MAJOR_CLEAN_H = {"DISP": 2.0, "GRAN": 4.0, "DRY": 3.5, "BLEND": 3.0, "COMP": 6.0, "COAT": 5.0, "PACK": 4.0}
POTENT_FACTOR = 1.8                                  # major clean after a potent product
DRY_HANDLING_H = 0.5                                 # loading, unloading, transfer around drying
# breakdowns: exponential time between failures (operating hours), lognormal repair (median, sigma)
MTBF_H = {"DISP": 400, "GRAN": 250, "DRY": 300, "BLEND": 400, "COAT": 200, "PACK": 150}
MTTR = {"DISP": (2.0, 0.5), "GRAN": (3.0, 0.5), "DRY": (3.0, 0.5), "BLEND": (2.0, 0.5),
        "COAT": (4.0, 0.6), "PACK": (2.5, 0.6), "COMP": (6.0, 0.6)}
PRESS_WEIBULL = (2.2, 260.0)                         # shape, scale (operating hours since PM)
PM_H = 4.0
QC_WORK_H, QC_LAG_H = 6.0, 30.0                      # analyst hands-on hours; fixed lag (micro etc.)

# economics per simulated quarter (13 weeks), illustrative
COST = {
    "shift_week": {2: 0.0, 3: 38e3, "24/7": 70e3},   # incremental over the 2-shift base, per week
    "analyst_week": 1.7e3,
    "coater2_quarter": 45e3,                         # capex amortisation plus upkeep
    "twin_dryer_quarter": 18e3,                      # sensors, software, model upkeep
    "pm_event": 1.5e3,
    "breakdown_event": 6e3,
    "holding_rate_quarter": 0.05,                    # 20% per year on cycle stock
}
WARMUP_H = 2 * WEEK
HORIZON_H = 13 * WEEK


@dataclass
class Config:
    shifts: object = 2            # 2, 3 or "24/7"
    qc_analysts: int = 3
    campaign: int = 2             # batches of the highest-demand product per campaign
    conwip: int = 6               # max batches between dispensing start and packaging end
    pm_interval: float = 160.0    # press operating hours between preventive maintenance
    dryer: str = "fixed"          # end-point policy: fixed, temp, twin
    coaters: int = 1

    def key(self):
        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------------------------
# Calendars
# --------------------------------------------------------------------------------------------
def is_open(t, shifts):
    """Production calendar. Week starts Monday 06:00 (t = 0)."""
    if shifts == "24/7":
        return True
    tw = t % WEEK
    day, hour = int(tw // 24), tw % 24
    if day >= 5:
        return False
    return hour < (16.0 if shifts == 2 else 24.0)


def next_open(t, shifts, step=0.25):
    while not is_open(t, shifts):
        t = (np.floor(t / step) + 1) * step
    return t


def qc_open(t):
    """QC analysts: Monday to Friday, 08:00 to 17:00 (t = 0 is 06:00)."""
    tw = t % WEEK
    day, hour = int(tw // 24), tw % 24
    return day < 5 and 2.0 <= hour < 11.0


# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------
class Machine:
    def __init__(self, name):
        self.name = name
        self.last_product = None
        self.age = 0.0            # operating hours since PM (presses) or since repair
        self.busy = 0.0
        self.cleaning = 0.0
        self.down = 0.0


class PlantSim:
    def __init__(self, cfg, seed, dryer_samples):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.dryer_samples = np.asarray(dryer_samples[cfg.dryer]) / 60.0  # minutes -> hours
        units = dict(EQUIP_UNITS)
        units["COAT"] = cfg.coaters
        self.stores = {}
        for eq, n in units.items():
            st = simpy.Store(self.env, capacity=n)
            st.items = [Machine(f"{eq}{i + 1}") for i in range(n)]
            self.stores[eq] = st
        self.machines = {eq: list(st.items) for eq, st in self.stores.items()}
        self.wip = simpy.Container(self.env, capacity=cfg.conwip, init=cfg.conwip)
        self.qc = simpy.Resource(self.env, capacity=cfg.qc_analysts)
        self.released = []
        self.counts = {"changeovers": 0, "pm": 0, "breakdowns": 0}
        self.sequence = self._build_cycle()

    def _build_cycle(self):
        top = max(DEMAND_SHARE.values())
        cyc = []
        for p in PRODUCTS:
            n = max(1, int(round(self.cfg.campaign * DEMAND_SHARE[p] / top)))
            cyc += [p] * n
        return cyc

    def _lognormal(self, median, cv):
        sigma = np.sqrt(np.log(1 + cv * cv))
        return float(median * np.exp(self.rng.normal(0.0, sigma)))

    def _duration(self, eq, product, median, cv):
        if eq == "DRY":
            return float(self.rng.choice(self.dryer_samples)) + DRY_HANDLING_H
        if eq == "COMP":
            return self._lognormal(COMP_H[product], cv)
        return self._lognormal(median, cv)

    def _clean_time(self, eq, m, product):
        if m.last_product is None:
            return 0.0
        if m.last_product == product:
            return MINOR_CLEAN_H[eq]
        self.counts["changeovers"] += 1
        f = POTENT_FACTOR if m.last_product in POTENT else 1.0
        return MAJOR_CLEAN_H[eq] * f

    def _wait_open(self):
        t = next_open(self.env.now, self.cfg.shifts)
        if t > self.env.now:
            yield self.env.timeout(t - self.env.now)

    def _run_step(self, eq, m, product, dur):
        """Process time with possible breakdowns (exponential, or Weibull age-based for presses)."""
        remaining = dur
        while remaining > 1e-9:
            if eq == "COMP":
                k, lam = PRESS_WEIBULL
                u = self.rng.random()
                # residual life given current age (Weibull)
                t_fail = lam * ((m.age / lam) ** k - np.log(u)) ** (1.0 / k) - m.age
            else:
                t_fail = self.rng.exponential(MTBF_H[eq])
            run = min(remaining, t_fail)
            yield self.env.timeout(run)
            m.busy += run
            m.age += run
            remaining -= run
            if remaining > 1e-9:
                self.counts["breakdowns"] += 1
                rep = float(MTTR[eq][0] * np.exp(self.rng.normal(0.0, MTTR[eq][1])))
                yield self.env.timeout(rep)
                m.down += rep
                if eq == "COMP":
                    m.age = 0.3 * m.age   # minimal-to-imperfect repair

    def batch(self, bid, product):
        env = self.env
        start = env.now
        for eq, median, cv in STEPS:
            if eq == "COAT" and not COATED[product]:
                continue
            m = yield self.stores[eq].get()
            yield from self._wait_open()
            if eq == "COMP" and m.age >= self.cfg.pm_interval:
                yield env.timeout(PM_H)
                m.down += PM_H
                m.age = 0.0
                self.counts["pm"] += 1
            c = self._clean_time(eq, m, product)
            if c > 0:
                yield env.timeout(c)
                m.cleaning += c
                yield from self._wait_open()
            dur = self._duration(eq, product, median, cv)
            yield from self._run_step(eq, m, product, dur)
            m.last_product = product
            yield self.stores[eq].put(m)
        yield self.wip.put(1)
        pack_end = env.now
        # QC: fixed lag (micro, documentation) in parallel with analyst work in lab hours
        lag = env.timeout(QC_LAG_H)
        work_left = QC_WORK_H
        with self.qc.request() as req:
            yield req
            while work_left > 1e-9:
                if qc_open(env.now):
                    yield env.timeout(0.25)
                    work_left -= 0.25
                else:
                    yield env.timeout(0.25)
        yield lag
        self.released.append({"id": bid, "product": product, "start": start,
                              "pack_end": pack_end, "release": env.now})

    def dispatcher(self):
        i = 0
        while True:
            yield self.wip.get(1)
            yield from self._wait_open()
            product = self.sequence[i % len(self.sequence)]
            self.env.process(self.batch(i, product))
            i += 1

    def run(self):
        self.env.process(self.dispatcher())
        self.env.run(until=WARMUP_H + HORIZON_H)
        return self.kpis()

    def kpis(self):
        cfg = self.cfg
        win = [r for r in self.released if r["release"] >= WARMUP_H]
        n = len(win)
        by_p = {p: sum(1 for r in win if r["product"] == p) for p in PRODUCTS}
        sold = {p: min(by_p[p], DEMAND_Q[p]) for p in PRODUCTS}
        contrib = sum(CONTRIB_EUR[p] * sold[p] for p in PRODUCTS)
        service = sum(sold.values()) / sum(DEMAND_Q.values())
        lead = [r["release"] - r["start"] for r in win]
        weeks = HORIZON_H / WEEK
        cost = (COST["shift_week"][cfg.shifts] * weeks + COST["analyst_week"] * cfg.qc_analysts * weeks
                + (COST["coater2_quarter"] if cfg.coaters > 1 else 0.0)
                + (COST["twin_dryer_quarter"] if cfg.dryer == "twin" else 0.0)
                + COST["pm_event"] * self.counts["pm"] * HORIZON_H / (WARMUP_H + HORIZON_H)
                + COST["breakdown_event"] * self.counts["breakdowns"] * HORIZON_H / (WARMUP_H + HORIZON_H))
        cycle_stock = sum(max(1, round(cfg.campaign * DEMAND_SHARE[p] / max(DEMAND_SHARE.values()))) / 2
                          for p in PRODUCTS) * BATCH_VALUE_EUR
        holding = cycle_stock * COST["holding_rate_quarter"]
        total_h = WARMUP_H + HORIZON_H
        util = {eq: sum(m.busy for m in ms) / (len(ms) * total_h) for eq, ms in self.machines.items()}
        clean = {eq: sum(m.cleaning for m in ms) / (len(ms) * total_h) for eq, ms in self.machines.items()}
        return {
            "released": n, "service_level": service, **{f"rel_{p}": v for p, v in by_p.items()},
            "lead_time_h_mean": float(np.mean(lead)) if lead else float("nan"),
            "lead_time_h_p90": float(np.percentile(lead, 90)) if lead else float("nan"),
            "equipment_changeovers": self.counts["changeovers"], "pm_events": self.counts["pm"],
            "breakdowns": self.counts["breakdowns"],
            **{f"util_{eq}": u for eq, u in util.items()},
            **{f"clean_{eq}": c for eq, c in clean.items()},
            "contribution_eur": contrib, "cost_eur": cost + holding, "holding_eur": holding,
            "margin_eur": contrib - cost - holding,
        }


def simulate(cfg, seed, dryer_samples):
    return PlantSim(cfg, seed, dryer_samples).run()
