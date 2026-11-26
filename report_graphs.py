"""Report graphs for the dual-hormone (insulin + glucagon) simulator.

Two steps, so a run made BEFORE a model change stays on disk and can be overlaid later:

    python report_graphs.py run --tag glucagon_v0_linear          # simulate, save data, draw figures
    python report_graphs.py run --tag glucagon_v0_linear --quick  # 2 patients, fewer doses (smoke test)
    python report_graphs.py plot --tag glucagon_v0_linear         # redraw from saved CSVs only
    python report_graphs.py compare --tags glucagon_v0_linear glucagon_v1_saturating

Output goes to results/<tag>/ (git-ignored, throwaway). To keep a run as a baseline, copy it by hand:
    cp -r results/<tag> baselines/<tag>       then commit baselines/<tag>
plot and compare look for a tag in results/ first, then in baselines/. Each tag folder holds:
    manifest.json   commit, dirty flag, command, seed, patients, glucagon parameter row, versions
    *.csv           the data behind every figure (the "table view")
    figures/        the PNGs

Protocol for the hypoglycemia experiments (mirrors the clinical protocol in Wendt 2017 / Furio 2023):
correction bolus (repeated if BG stops falling above 70) -> wait until true BG <= 70 mg/dl
-> glucagon (or nothing) at t = 0 -> follow 240 min.
Fasted, no meals, basal insulin only. The pre-dose phase is identical for every dose, so a 0 ug run
is an exact control.

Figures:
    01 dose response   02 PK/PD cascade   03 peak vs dose   04 insulin gate
    05 stress test     06 rescue trace    07 population fan 08 phase portrait

Internal states are read in ONE place (snapshot()). When the glucagon model changes, update it there.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no windows; must be set before pyplot is imported (run_sim imports it)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from run_sim import DEFAULT_MEALS, RescueController, run_dual, summarize  # noqa: E402
from simglucose.actuator.pump import InsulinPump, PumpConfig, PumpModel  # noqa: E402
from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST, BBController  # noqa: E402
from simglucose.patient.t1dpatient import (  # noqa: E402
    PATIENT_PARA_FILE,
    T1DPatient,
    load_glucagon_params,
)
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.dual_env import DualAction, DualHormoneEnv  # noqa: E402
from simglucose.simulation.scenario import CustomScenario  # noqa: E402

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"       # throwaway output of every run (git-ignored)
BASELINES = REPO / "baselines"   # runs you copied here by hand to keep

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------
START = datetime(2024, 1, 1, 0, 0)
SAMPLE = 3                 # min, Dexcom CGM sample time = env step
POST_MIN = 240             # follow-up after the glucagon dose
PRE_MAX_MIN = 18 * 60      # give up if BG never reaches the trigger
MAX_BOLUSES = 6            # repeat the correction if BG stops falling above the trigger
STALL_STEPS = 30           # steps (90 min) after a bolus before it counts as "stopped falling"
TRIGGER = 70.0             # mg/dl, true BG that starts the protocol
TARGET_NADIR = 54.0        # mg/dl, the correction bolus aims here (Wendt 2017)
DEFAULT_PANEL = ["adolescent#001", "adolescent#002", "adult#001", "adult#002", "child#001", "child#002"]
QUICK_PANEL = ["adolescent#001", "adult#001"]
DOSES = list(range(0, 501, 50))
QUICK_DOSES = [0, 100, 200, 300, 500]
MULTS = [1.0, 2.0, 3.0]
GATE_DOSE = 200
STRESS_HOURS = 24
PHASE_MAX_MIN = 90         # phase portrait stops once plasma glucagon is back near zero

# Approximate peaks read off Furio-Novejarque 2023 Fig. 7 (mean of 8 patients, +/- 10 mg/dl, ~45-50 min).
# NOT tabulated data. Replace with digitized points if you need a curve.
REFERENCE_PEAKS = {100: (47.5, 100.0), 200: (47.5, 128.0), 300: (47.5, 148.0)}

# --------------------------------------------------------------------------
# Style (reference palette from the dataviz skill; ordinal ramp and slots 1-3 passed validate_palette.js)
# --------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRIT = "#d03b3b"
DOSE_COLORS = {0: MUTED, 100: "#86b6ef", 200: "#2a78d6", 300: "#184f95"}
MULT_COLORS = {1.0: "#86b6ef", 2.0: "#2a78d6", 3.0: "#184f95"}
TAG_COLORS = [BLUE, ORANGE, AQUA]
TAG_LS = ["--", "-", "-."]  # first tag listed is the baseline (dashed) when several are overlaid


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "text.color": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 9,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 150, "font.family": "sans-serif",
    })


def _ls(tags, i):
    return "-" if len(tags) == 1 else TAG_LS[i]


def _save(fig, out_dir, name):
    path = Path(out_dir) / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def _note(fig, text):
    """Wrapped caption at the bottom; remembers how much room _tight() must leave for it."""
    wrapped = textwrap.fill(text, width=int(fig.get_figwidth() * 17))
    fig.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig._note_bottom = (0.1 + 0.14 * (wrapped.count("\n") + 1)) / fig.get_figheight()


def _tight(fig, top=1.0):
    fig.tight_layout(rect=(0, getattr(fig, "_note_bottom", 0.0), 1, top))


# --------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------
def snapshot(env):
    """Read internal states for plotting. The ONLY place that touches private patient attributes.

    Legacy model: 3 glucagon states (depot g0, plasma g1, delayed effect g2) in patient.state[13:16].
    plasma_glucagon_pg_ml is NaN until the planned PK change provides it (GLUCAGON_PLAN.md step 2).
    """
    p = env.patient
    x = p.state
    prm = p._params
    gp = getattr(p, "_gparams", None)
    g0, g1, g2 = float(x[13]), float(x[14]), float(x[15])
    kgn = float(gp["kgn"]) if gp is not None and "kgn" in gp else float("nan")
    egp_gl = kgn * g2
    egp = float(prm.kp1 - prm.kp2 * x[3] - prm.kp3 * x[8]) + egp_gl
    return {
        "g0": g0, "g1": g1, "g2": g2,
        "egp": max(egp, 0.0), "egp_gl": egp_gl,
        "insulin_mU_l": float(x[5] / prm.Vi / 6.0),  # pmol/l -> mU/l (1 U = 6000 pmol)
        "plasma_glucagon_pg_ml": float("nan"),
    }


def _build_env(name, seed):
    patient = T1DPatient.withName(name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    insulin_pump = InsulinPump.withName("Insulet")
    glucagon_pump = PumpModel(PumpConfig(seed=seed))  # ideal pump: no clipping up to 500 ug
    scenario = CustomScenario(start_time=START, scenario=[])  # no meals
    return DualHormoneEnv(patient, sensor, insulin_pump, glucagon_pump, scenario)


def _correction_factor(name):
    quest = pd.read_csv(CONTROL_QUEST)
    row = quest[quest.Name == name]
    return float(row.CF.iloc[0]) if len(row) else 50.0  # mg/dl per U


def run_hypo_protocol(name, dose_ug, bolus_mult=1.0, seed=1, post_min=POST_MIN):
    """Correction bolus, wait for BG <= TRIGGER, dose at t=0, follow post_min. Returns a DataFrame or None."""
    env = _build_env(name, seed)
    bb = BBController()
    obs, reward, done, info = env.reset()
    snaps = [snapshot(env)]
    bg0, cf = info["bg"], _correction_factor(name)
    bgs = [bg0]                       # true BG per step, to detect "stopped falling"
    first_u = max(bolus_mult * (bg0 - TARGET_NADIR) / cf, 0.0)
    bolus_u, n_bolus, last_bolus = first_u, 1, 0  # bolus_u = cumulative insulin given before the dose

    steps, k0 = 0, None
    n_post = post_min // SAMPLE
    while True:
        if k0 is None and info["bg"] <= TRIGGER:
            k0 = steps
        pending = first_u if steps == 0 else 0.0
        if (k0 is None and steps > 0 and n_bolus < MAX_BOLUSES and steps - last_bolus >= STALL_STEPS
                and bgs[-1] >= bgs[-2]):
            # BG stopped falling above the trigger: repeat the correction, sized from the sensitivity observed so far
            drop = bg0 - min(bgs)
            cf_eff = drop / bolus_u if drop > 0.5 and bolus_u > 0 else cf
            pending = max(bolus_mult * (info["bg"] - TARGET_NADIR) / cf_eff, 0.2)
            bolus_u += pending
            n_bolus += 1
            last_bolus = steps
        base = bb.policy(obs, reward, done, **info)  # meal = 0, so basal only
        bolus = pending / SAMPLE  # U/min held for one sample
        glucagon = dose_ug / SAMPLE if (k0 is not None and steps == k0) else 0.0  # ug/min held for one sample
        obs, reward, done, info = env.step(DualAction(base.basal, bolus, glucagon))
        steps += 1
        bgs.append(info["bg"])
        snaps.append(snapshot(env))
        if done:  # BG left the simulator's safe range
            break
        if k0 is not None and steps >= k0 + n_post:
            break
        if k0 is None and steps >= PRE_MAX_MIN // SAMPLE:
            break
    if k0 is None:
        raise RuntimeError(f"BG never reached {TRIGGER:.0f} mg/dl: start {bg0:.0f}, lowest {min(bgs):.0f}, "
                           f"{n_bolus} bolus(es) totalling {bolus_u:.1f} U, correction factor {cf:.1f} mg/dl/U")

    hist = env.show_history().reset_index(drop=True)
    full = pd.concat([hist, pd.DataFrame(snaps)], axis=1)
    seg = full.iloc[k0:k0 + n_post + 1].copy()
    seg["t_min"] = np.arange(len(seg)) * SAMPLE
    seg = seg.set_index("t_min").reindex(np.arange(0, post_min + 1, SAMPLE))
    seg.index.name = "t_min"
    seg = seg.reset_index()
    seg["patient"], seg["dose_ug"], seg["bolus_mult"] = name, dose_ug, bolus_mult
    seg["bolus_u"] = bolus_u
    seg["trigger_min"] = k0 * SAMPLE
    seg["insulin_at_dose_mU_l"] = snaps[k0]["insulin_mU_l"]
    return seg


def _curve_job(args):
    name, dose, mult, seed = args
    try:
        return run_hypo_protocol(name, dose, mult, seed)
    except Exception as exc:  # keep one bad patient from killing the whole run
        print(f"  ! {name} dose={dose} mult={mult}: {exc}")
        return None


def _stress_job(args):
    name, mult, arm, seed = args
    meals = [(float(h), g) for h, g in DEFAULT_MEALS]
    threshold = 75.0 if arm == "glucagon" else -1.0  # -1 never fires: insulin-only reference
    ctrl = RescueController(bolus_mult=mult, threshold=threshold, dose_ug=150.0)
    df = run_dual(name, STRESS_HOURS, meals, seed, PumpConfig(max_flow=100.0, seed=seed), ctrl)
    st = summarize(df)
    dt = (df.index[1] - df.index[0]).total_seconds() / 60
    return {
        "patient": name, "bolus_mult": mult, "arm": arm, "TIR": st["TIR"], "TBR70": st["TBR"],
        "TBR54": 100 * float((df["BG"] < 54).mean()), "TAR": st["TAR"], "nadir": float(df["BG"].min()),
        "glucagon_ug": float(df["glucagon"].sum() * dt), "rescues": ctrl.rescues,
    }


def _concat(results, what):
    ok = [r for r in results if r is not None]
    if not ok:
        sys.exit(f"No successful runs for {what}. See the '!' messages above for the reason.")
    return pd.concat(ok)


def _map(fn, items, jobs):
    if jobs <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items, chunksize=1))


def all_patient_names():
    return pd.read_csv(PATIENT_PARA_FILE).Name.tolist()


def _trapz(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = ~np.isnan(y)
    y, x = y[m], x[m]
    if len(y) < 2:
        return float("nan")
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def curve_metrics(curves):
    """Per patient and dose: peak rise, time to peak and incremental AUC, all measured against the
    matched 0 ug run of the same patient. (Measured from the starting glucose instead, the late recovery
    as insulin wears off counts as a 'response' even with no glucagon.) Also minimum BG and minutes < 70."""
    rows = []
    ctrl = {p: g.set_index("t_min")["BG"] for p, g in curves[curves.dose_ug == 0].groupby("patient")}
    for (p, d), g in curves.groupby(["patient", "dose_ug"]):
        bg = g.sort_values("t_min").set_index("t_min")["BG"]
        if bg.empty or np.isnan(bg.iloc[0]):
            continue
        net = (bg - ctrl[p]) if p in ctrl else (bg - bg.iloc[0])
        net = net.dropna()
        if net.empty:
            continue
        pos = net.clip(lower=0)
        rows.append({"patient": p, "dose_ug": d, "bg0": float(bg.iloc[0]), "peak_dbg": float(net.max()),
                     "t_peak": float(net.idxmax()), "peak_bg": float(bg.max()),
                     "iauc": _trapz(pos.to_numpy(), pos.index.to_numpy()), "min_bg": float(bg.min()),
                     "min_below_70": float((bg < 70).sum() * SAMPLE)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_manifest(out, args, patients, doses, skipped):
    try:
        gparams = {k: (float(v) if not isinstance(v, str) else v) for k, v in load_glucagon_params().to_dict().items()}
    except Exception:
        gparams = None
    manifest = {
        "tag": args.tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "seed": args.seed, "quick": args.quick,
        "patients": patients, "doses_ug": doses, "bolus_mults": MULTS, "gate_dose_ug": GATE_DOSE,
        "trigger_mg_dl": TRIGGER, "target_nadir_mg_dl": TARGET_NADIR, "sample_min": SAMPLE,
        "stress_hours": STRESS_HOURS, "skipped_patients": skipped,
        "glucagon_params_default_row": gparams,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["git_dirty"]:
        print("  note: git tree is dirty; commit before treating this run as a reproducible baseline.")


def run_experiments(args):
    if (BASELINES / args.tag).exists():
        sys.exit(f"'{args.tag}' already exists in baselines/. Pick a new --tag so a kept baseline is never shadowed.")
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)  # results/ is throwaway: re-running a tag overwrites it

    patients = args.patients.split(",") if args.patients else (QUICK_PANEL if args.quick else DEFAULT_PANEL)
    doses = QUICK_DOSES if args.quick else DOSES
    pop_patients = patients if args.quick else all_patient_names()
    seed, jobs = args.seed, args.jobs

    print(f"[1/5] dose curves: {len(patients)} patients x {len(doses)} doses")
    items = [(p, d, 1.0, seed) for p in patients for d in doses]
    res = _map(_curve_job, items, jobs)
    skipped = sorted({it[0] for it, r in zip(items, res) if r is None})
    if skipped:
        print(f"  skipped patients (see '!' messages above): {skipped}")
    _concat(res, "dose curves").to_csv(out / "dose_curves.csv", index=False)

    print(f"[2/5] insulin dependence: {len(patients)} patients x {len(MULTS)} bolus multipliers x 2")
    items = [(p, d, m, seed) for p in patients for m in MULTS for d in (0, GATE_DOSE)]
    _concat(_map(_curve_job, items, jobs), "insulin dependence").to_csv(out / "insulin_dependence.csv", index=False)

    print(f"[3/5] population: {len(pop_patients)} patients, doses 0 and {GATE_DOSE}")
    items = [(p, d, 1.0, seed) for p in pop_patients for d in (0, GATE_DOSE)]
    pop = _concat(_map(_curve_job, items, jobs), "population")
    pop[["patient", "dose_ug", "t_min", "BG"]].to_csv(out / "population.csv", index=False)

    if not args.skip_stress:
        print(f"[4/5] closed-loop stress test: {len(patients)} patients x {len(MULTS)} multipliers x 2 arms, {STRESS_HOURS} h")
        items = [(p, m, a, seed) for p in patients for m in MULTS for a in ("insulin_only", "glucagon")]
        pd.DataFrame(_map(_stress_job, items, jobs)).to_csv(out / "stress.csv", index=False)

        print("[5/5] rescue trace")
        ctrl = RescueController(bolus_mult=2.0, threshold=75.0, dose_ug=150.0)
        meals = [(float(h), g) for h, g in DEFAULT_MEALS]
        df = run_dual(patients[0], STRESS_HOURS, meals, seed, PumpConfig(max_flow=100.0, seed=seed), ctrl)
        df["patient"] = patients[0]
        df.to_csv(out / "rescue_trace.csv")

    write_manifest(out, args, patients, doses, skipped)
    print("drawing figures")
    plot_all([args.tag], out)
    print(f"done. Throwaway output is in {out.relative_to(REPO)}/. To keep it as a baseline:\n"
          f"  cp -r results/{args.tag} baselines/{args.tag}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def tag_dir(tag):
    """Where a tag's data lives: results/ (new runs) first, then baselines/ (kept runs)."""
    for root in (RESULTS, BASELINES):
        if (root / tag).exists():
            return root / tag
    sys.exit(f"no data for tag '{tag}' in {RESULTS} or {BASELINES}")


def load_tag(tag):
    d = tag_dir(tag)
    data = {}
    for name in ("dose_curves", "insulin_dependence", "population", "stress"):
        f = d / f"{name}.csv"
        data[name] = pd.read_csv(f) if f.exists() else None
    f = d / "rescue_trace.csv"
    data["rescue_trace"] = pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None
    return data


def _tag_handles(tags):
    if len(tags) < 2:
        return []
    return [Line2D([], [], color=INK2, lw=1.8, ls=TAG_LS[i], label=t) for i, t in enumerate(tags)]


def _full(piv):
    """True where every patient still has data (a run ends if glucose falls below 10 mg/dl).
    Averaging over a shrinking set of patients makes the mean jump, so lines stop at the first dropout."""
    return piv.count(axis=1) == piv.shape[1]


def _mean_sd(df, value="BG"):
    piv = df.pivot_table(index="t_min", columns="patient", values=value)
    full = _full(piv)
    return piv.mean(axis=1).where(full), piv.std(axis=1).where(full)


def fig_dose_response(data, tags, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for ti, tag in enumerate(tags):
        c = data[tag]["dose_curves"]
        for d in (0, 100, 200, 300):
            g = c[c.dose_ug == d]
            if g.empty:
                continue
            m, s = _mean_sd(g)
            ax.plot(m.index, m.values, color=DOSE_COLORS[d], ls=_ls(tags, ti))
            if len(tags) == 1:
                ax.fill_between(m.index, (m - s).values, (m + s).values, color=DOSE_COLORS[d], alpha=0.12, lw=0)
    for d, (t_ref, bg_ref) in REFERENCE_PEAKS.items():
        ax.errorbar(t_ref, bg_ref, xerr=5, yerr=10, fmt="o", ms=6, mfc=SURFACE, mec=DOSE_COLORS[d], mew=1.5,
                    ecolor=DOSE_COLORS[d], elinewidth=1, capsize=3)
    ax.axhline(70, color=CRIT, lw=0.8)
    ax.text(238, 71.5, "70 mg/dl", ha="right", va="bottom", fontsize=7.5, color=INK2)
    handles = [Line2D([], [], color=DOSE_COLORS[d], lw=1.8, label="saline / no glucagon" if d == 0 else f"{d} µg")
               for d in (0, 100, 200, 300)]
    handles.append(Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=MUTED, mew=1.5,
                          label="Furió 2023 Fig. 7 peaks (approx.)"))
    ax.legend(handles=handles + _tag_handles(tags), loc="upper right", ncol=2)
    ax.set_xlim(0, 240)
    ax.set_ylim(40, 200)
    ax.set_xlabel("Minutes after glucagon dose")
    ax.set_ylabel("Blood glucose (mg/dl)")
    n = data[tags[0]]["dose_curves"].patient.nunique()
    ax.set_title("Glucose response to a subcutaneous glucagon dose after insulin-induced hypoglycemia", loc="left")
    _note(fig, f"n = {n} virtual patients" + (", mean ± SD" if len(tags) == 1 else ", mean") +
          ". Reference markers are approximate peaks read off the figure (±10 mg/dl), not tabulated data.")
    _tight(fig)
    _save(fig, out, "01_dose_response.png")


def fig_cascade(curves, out, suffix="", dose=GATE_DOSE):
    doses = sorted(curves.dose_ug.unique())
    dose = dose if dose in doses else doses[-1]
    patient = sorted(curves.patient.unique())[0]
    g = curves[(curves.patient == patient) & (curves.dose_ug == dose)].sort_values("t_min")
    t = g.t_min.to_numpy()
    has_conc = bool(g.plasma_glucagon_pg_ml.notna().any())
    n = 4 + int(has_conc)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 1.9 * n + 0.9), sharex=True)
    ax = iter(axes)

    a = next(ax)  # pump flow
    a.step(t, g.glucagon.fillna(0), where="post", color=BLUE)
    a.set_ylabel("Pump flow\n(µg/min)")
    a.set_ylim(bottom=0)

    a = next(ax)  # PK states
    a.plot(t, g.g0, color=BLUE, label="Depot (subcutaneous)")
    a.plot(t, g.g1, color=ORANGE, label="Plasma, above basal")
    a.set_ylabel("Glucagon\n(µg/kg)")
    a.legend(loc="upper right")

    if has_conc:
        a = next(ax)
        a.plot(t, g.plasma_glucagon_pg_ml, color=BLUE)
        a.set_ylabel("Plasma glucagon\n(pg/ml)")

    a = next(ax)  # EGP
    a.plot(t, g.egp, color=BLUE, label="Total EGP")
    a.plot(t, g.egp_gl, color=ORANGE, label="Glucagon contribution")
    a.set_ylabel("EGP\n(mg/kg/min)")
    a.legend(loc="upper right")

    a = next(ax)  # glucose
    a.plot(t, g.BG, color=BLUE)
    a.axhline(70, color=CRIT, lw=0.8)
    a.set_ylabel("Blood glucose\n(mg/dl)")
    a.set_xlabel("Minutes after glucagon dose")
    a.set_xlim(0, 240)

    t_pl = t[int(np.nanargmax(g.g1.to_numpy(float)))]
    t_bg = t[int(np.nanargmax(g.BG.to_numpy(float)))]
    for axx in axes:
        for x in (t_pl, t_bg):
            axx.axvline(x, color=MUTED, lw=0.8, ls=":")
    axes[0].text(t_pl, 1.03, f"plasma peak {t_pl:.0f} min", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color=INK2)
    axes[0].text(t_bg, 1.03, f"glucose peak {t_bg:.0f} min", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color=INK2)
    fig.suptitle(f"Lag chain for a {dose} µg dose ({patient})", x=0.01, ha="left", fontsize=10, fontweight="bold")
    _note(fig, "Legacy model: plasma glucagon is in µg/kg (no pg/ml until the PK change). EGP is approximate, from end-of-step states.")
    _tight(fig, 0.97)
    _save(fig, out, f"02_pk_pd_cascade{suffix}.png")


def fig_peak_vs_dose(data, tags, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    specs = [("peak_dbg", "Peak glucose rise vs 0 µg run (mg/dl)"), ("t_peak", "Time to peak (min)"),
             ("iauc", "Incremental AUC vs 0 µg run (mg/dl·min)")]
    first = None
    for ti, tag in enumerate(tags):
        m = curve_metrics(data[tag]["dose_curves"])
        for ax, (col, _label) in zip(axes, specs):
            mm = m[m.dose_ug > 0] if col == "t_peak" else m
            agg = mm.groupby("dose_ug")[col].agg(["mean", "min", "max"])
            ax.errorbar(agg.index, agg["mean"], yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                        color=TAG_COLORS[ti], ls=_ls(tags, ti), marker="o", ms=5, mec=SURFACE, mew=1.5,
                        elinewidth=1, capsize=0, label=tag)
        if first is None:
            first = m
    if first is not None and 100 in set(first.dose_ug):
        base = first[first.dose_ug == 100]
        for ax, col in zip((axes[0], axes[2]), ("peak_dbg", "iauc")):
            b = base[col].mean()
            xs = np.array(sorted(first.dose_ug.unique()))
            ax.plot(xs, b * xs / 100.0, color=MUTED, lw=1, ls=":", label="proportional to 100 µg response")
    for ax, (_c, label) in zip(axes, specs):
        ax.set_xlabel("Glucagon dose (µg)")
        ax.set_ylabel(label)
        ax.set_xlim(left=0)
    axes[0].legend(loc="upper left")
    fig.suptitle("Dose dependence of the glucagon response (mean, bars = min to max across patients)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    _note(fig, "Rise and area are measured against the matched 0 µg run of each patient. Time to peak excludes 0 µg. "
               "The dotted line extrapolates the first tag's 100 µg response in proportion to dose.")
    _tight(fig, 0.95)
    _save(fig, out, "03_peak_vs_dose.png")


def fig_insulin_gate(data, tags, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    mult_handles = []
    for ti, tag in enumerate(tags):
        ins = data[tag]["insulin_dependence"]
        for m in MULTS:
            dose = ins[(ins.dose_ug == GATE_DOSE) & (ins.bolus_mult == m)]
            ctrl = ins[(ins.dose_ug == 0) & (ins.bolus_mult == m)]
            if dose.empty or ctrl.empty:
                continue
            pd_ = dose.pivot_table(index="t_min", columns="patient", values="BG")
            pc_ = ctrl.pivot_table(index="t_min", columns="patient", values="BG")
            delta = pd_ - pc_
            pm = pd_.mean(axis=1).where(_full(pd_))
            dm = delta.mean(axis=1).where(_full(delta))
            a1.plot(pm.index, pm, color=MULT_COLORS[m], ls=_ls(tags, ti))
            a2.plot(dm.index, dm, color=MULT_COLORS[m], ls=_ls(tags, ti))
            if len(tags) == 1:
                s = delta.std(axis=1).where(_full(delta))
                a2.fill_between(dm.index, dm - s, dm + s, color=MULT_COLORS[m], alpha=0.12, lw=0)
            if ti == 0:
                i_mu = dose.groupby("patient").insulin_at_dose_mU_l.first().mean()
                mult_handles.append(Line2D([], [], color=MULT_COLORS[m], lw=1.8,
                                           label=f"{m:g}× correction bolus (plasma insulin at dose ≈ {i_mu:.0f} mU/l)"))
    a2.axhline(0, color=AXIS, lw=0.8)
    a1.axhline(70, color=CRIT, lw=0.8)
    a1.set_title(f"Glucose after {GATE_DOSE} µg", loc="left")
    a2.set_title(f"Glucagon effect: {GATE_DOSE} µg run minus matched 0 µg run", loc="left")
    for a in (a1, a2):
        a.set_xlabel("Minutes after glucagon dose")
        a.set_xlim(0, 240)
    a1.set_ylabel("Blood glucose (mg/dl)")
    a2.set_ylabel("Glucose difference (mg/dl)")
    a1.legend(handles=mult_handles + _tag_handles(tags), loc="upper right")
    fig.suptitle("Does ambient insulin weaken glucagon's effect?", x=0.01, ha="left", fontsize=10, fontweight="bold")
    _note(fig, "Insulin level is set by the correction-bolus size, not clamped, so it is falling during the dose. "
               "The matched 0 µg run removes insulin's own glucose-lowering effect. "
               "Lines stop where any patient's run ended (glucose below 10 mg/dl).")
    _tight(fig, 0.95)
    _save(fig, out, "04_insulin_gate.png")


def fig_stress(data, tags, out):
    base = data[tags[0]]["stress"]
    series = [("Insulin only", MUTED, base[base.arm == "insulin_only"])]
    for ti, tag in enumerate(tags):
        s = data[tag]["stress"]
        series.append((f"{tag}" if len(tags) > 1 else "Insulin + glucagon rescue", TAG_COLORS[ti], s[s.arm == "glucagon"]))
    metrics = [("TBR70", "Time below 70 mg/dl (%)"), ("TBR54", "Time below 54 mg/dl (%)"),
               ("nadir", "Lowest glucose (mg/dl)"), ("glucagon_ug", "Glucagon delivered (µg/day)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    ns = len(series)
    w = min(0.16, 0.7 / ns)
    for ax, (col, label) in zip(axes, metrics):
        for si, (_name, color, df) in enumerate(series):
            if col == "glucagon_ug" and si == 0:
                continue  # the insulin-only arm delivers none
            off = (si - (ns - 1) / 2) * (w + 0.03)
            for xi, m in enumerate(MULTS):
                vals = df[df.bolus_mult == m][col].to_numpy(float)
                if len(vals) == 0:
                    continue
                ax.bar(xi + off, vals.mean(), width=w, color=color, ec=SURFACE, lw=1.5)
                ax.plot(np.full(len(vals), xi + off), vals, "o", ms=3.5, color=INK2, mec=SURFACE, mew=1, ls="")
        ax.set_xticks(range(len(MULTS)))
        ax.set_xticklabels([f"{m:g}×" for m in MULTS])
        ax.set_xlabel("Meal-bolus multiplier")
        ax.set_ylabel(label)
        if col == "nadir":
            ax.axhline(70, color=CRIT, lw=0.8)
    axes[0].legend(handles=[Line2D([], [], color=c, lw=6, label=n) for n, c, _ in series], loc="upper left")
    n = series[0][2].patient.nunique()
    fig.suptitle(f"Closed-loop stress test: {STRESS_HOURS} h, 3 meals, increasing insulin overdose (bars = mean, dots = patients)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    _note(fig, f"n = {n} virtual patients. Rescue: 150 µg when CGM < 75 mg/dl, 90 min cooldown. The insulin-only arm is taken from the first tag.")
    _tight(fig, 0.94)
    _save(fig, out, "05_stress_test.png")


def fig_rescue_trace(df, out, tag):
    t = (df.index - df.index[0]).total_seconds().to_numpy() / 3600
    dt = (df.index[1] - df.index[0]).total_seconds() / 60
    ts = t[1:]
    ins = df["insulin"].iloc[:-1].fillna(0).to_numpy() * 60
    carbs = df["CHO"].iloc[:-1].fillna(0).to_numpy() * dt
    gl = df["glucagon"].iloc[:-1].fillna(0).to_numpy()
    glc = df["glucagon_cmd"].iloc[:-1].fillna(0).to_numpy()
    fig, (a_bg, a_in, a_gl, a_ch) = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                                                  gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1]})
    a_bg.axhspan(70, 180, color=MUTED, alpha=0.12, lw=0)
    a_bg.plot(t, df["BG"], color=BLUE, label="True glucose")
    a_bg.plot(t, df["CGM"], color=ORANGE, lw=1.2, label="CGM")
    a_bg.axhline(70, color=CRIT, lw=0.8)
    a_bg.set_ylabel("Glucose (mg/dl)")
    a_bg.legend(loc="upper right", ncol=2)
    a_in.step(ts, ins, where="post", color=BLUE)
    a_in.set_yscale("symlog", linthresh=1)
    a_in.set_ylim(bottom=0)
    a_in.set_ylabel("Insulin (U/h)")
    a_gl.step(ts, glc, where="post", color=MUTED, ls="--", lw=1.2, label="Commanded")
    a_gl.step(ts, gl, where="post", color=ORANGE, label="Delivered")
    a_gl.set_ylim(bottom=0)
    a_gl.set_ylabel("Glucagon (µg/min)")
    a_gl.legend(loc="upper left", ncol=2)  # rescues happen late in the day, so the left side is empty
    a_ch.bar(ts, carbs, width=dt / 60, align="edge", color=ORANGE)
    a_ch.set_ylabel("Carbs (g/sample)")
    a_ch.set_xlabel("Hours")
    starts = [ts[i] for i in range(len(ts)) if gl[i] > 0 and (i == 0 or gl[i - 1] <= 0)]
    for s in starts:
        a_bg.axvline(s, color=MUTED, lw=0.8, ls=":")
    patient = df["patient"].iloc[0] if "patient" in df else ""
    fig.suptitle(f"Closed-loop rescue ({patient}, 2× meal bolus, {len(starts)} rescue(s)) [{tag}]",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, "06_rescue_trace.png")


def fig_population(data, tags, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n = None
    for ti, tag in enumerate(tags):
        pop = data[tag]["population"]
        g = pop[pop.dose_ug == GATE_DOSE].pivot_table(index="t_min", columns="patient", values="BG")
        n = g.shape[1]
        full = _full(g)
        med = g.median(axis=1).where(full)
        q25, q75 = g.quantile(0.25, axis=1).where(full), g.quantile(0.75, axis=1).where(full)
        color = TAG_COLORS[ti] if len(tags) > 1 else BLUE
        if len(tags) == 1:
            ax.fill_between(g.index, g.min(axis=1).where(full), g.max(axis=1).where(full),
                            color=color, alpha=0.10, lw=0)
            cp = pop[pop.dose_ug == 0].pivot_table(index="t_min", columns="patient", values="BG")
            ctrl = cp.median(axis=1).where(_full(cp))
            ax.plot(ctrl.index, ctrl.values, color=MUTED, lw=1.2, ls="--")
        ax.fill_between(g.index, q25, q75, color=color, alpha=0.25 if len(tags) == 1 else 0.15, lw=0)
        ax.plot(med.index, med.values, color=color, ls=_ls(tags, ti))
    ax.axhline(70, color=CRIT, lw=0.8)
    if len(tags) == 1:
        handles = [Line2D([], [], color=BLUE, lw=1.8, label=f"Median, {GATE_DOSE} µg"),
                   plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.25, lw=0, label="25th–75th percentile"),
                   plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.10, lw=0, label="Min–max"),
                   Line2D([], [], color=MUTED, lw=1.2, ls="--", label="Median, 0 µg")]
    else:
        handles = [Line2D([], [], color=TAG_COLORS[i], lw=1.8, ls=TAG_LS[i], label=f"{t} (median, IQR)")
                   for i, t in enumerate(tags)]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(0, 240)
    ax.set_ylim(30, 200)
    ax.set_xlabel("Minutes after glucagon dose")
    ax.set_ylabel("Blood glucose (mg/dl)")
    ax.set_title(f"Spread of the {GATE_DOSE} µg response across virtual patients", loc="left")
    _note(fig, f"n = {n} virtual patients. In the legacy model all patients share one glucagon parameter row, "
               "so spread comes from body weight and glucose-insulin parameters only.")
    _tight(fig)
    _save(fig, out, "07_population_fan.png")


def fig_phase(curves, out, suffix="", dose=GATE_DOSE):
    doses = sorted(curves.dose_ug.unique())
    dose = dose if dose in doses else doses[-1]
    patient = sorted(curves.patient.unique())[0]
    g = curves[(curves.patient == patient) & (curves.dose_ug == dose)].sort_values("t_min")
    g = g[g.t_min <= PHASE_MAX_MIN]  # later, glucagon is ~0 and glucose just retraces as insulin wears off
    has_conc = bool(g.plasma_glucagon_pg_ml.notna().any())
    x = (g.plasma_glucagon_pg_ml if has_conc else g.g1).to_numpy(float)
    y = g.BG.to_numpy(float)
    t = g.t_min.to_numpy()
    ok = ~(np.isnan(x) | np.isnan(y))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x[ok], y[ok], color=BLUE)
    for i in range(2, len(x) - 1, 6):
        if ok[i] and ok[i + 1] and np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) > 0:
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5, shrinkA=0, shrinkB=0))
    for ts in (0, 15, 30, 60, 90):
        idx = np.where(t == ts)[0]
        if len(idx) and ok[idx[0]]:
            i = idx[0]
            ax.plot(x[i], y[i], "o", ms=5, color=BLUE, mec=SURFACE, mew=1.5)
            ax.annotate(f"{ts} min", (x[i], y[i]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.grid(axis="x")
    ax.set_xlabel("Plasma glucagon (pg/ml)" if has_conc else "Plasma glucagon above basal (µg/kg)")
    ax.set_ylabel("Blood glucose (mg/dl)")
    ax.set_title(f"Glucose vs plasma glucagon, {dose} µg, first {PHASE_MAX_MIN} min ({patient})", loc="left")
    fig.tight_layout()
    _save(fig, out, f"08_phase_portrait{suffix}.png")


def plot_all(tags, out):
    _style()
    if len(tags) > 3:
        sys.exit("compare supports at most 3 tags (categorical slots 1-3 are the validated all-pairs set)")
    data = {t: load_tag(t) for t in tags}
    have = lambda key: all(data[t][key] is not None for t in tags)  # noqa: E731
    if have("dose_curves"):
        fig_dose_response(data, tags, out)
        fig_peak_vs_dose(data, tags, out)
        for t in tags:
            fig_cascade(data[t]["dose_curves"], out, "" if len(tags) == 1 else f"_{t}")
            fig_phase(data[t]["dose_curves"], out, "" if len(tags) == 1 else f"_{t}")
    if have("insulin_dependence"):
        fig_insulin_gate(data, tags, out)
    if have("stress"):
        fig_stress(data, tags, out)
    if have("population"):
        fig_population(data, tags, out)
    for t in tags:
        if data[t]["rescue_trace"] is not None:
            fig_rescue_trace(data[t]["rescue_trace"], out, t)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate, save data to baselines/<tag>/, draw figures")
    r.add_argument("--tag", required=True, help="name of this snapshot, e.g. glucagon_v0_linear")
    r.add_argument("--patients", help="comma-separated names (default: 6-patient panel)")
    r.add_argument("--quick", action="store_true", help="2 patients, fewer doses: smoke test")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    r.add_argument("--skip-stress", action="store_true", help="skip the 24 h closed-loop runs (slowest part)")

    p = sub.add_parser("plot", help="redraw figures for one tag from its saved CSVs")
    p.add_argument("--tag", required=True)

    c = sub.add_parser("compare", help="overlay 2-3 tags (first = baseline)")
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--out", help="output folder (default baselines/compare_<tags>)")

    args = ap.parse_args()
    if args.cmd == "run":
        run_experiments(args)
    elif args.cmd == "plot":
        plot_all([args.tag], RESULTS / args.tag)  # figures go to results/, even if the data is a kept baseline
    else:
        out = Path(args.out) if args.out else RESULTS / ("compare_" + "__".join(args.tags))
        plot_all(args.tags, out)


if __name__ == "__main__":
    main()
