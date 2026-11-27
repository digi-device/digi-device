"""Run one simulated patient and save a summary graph (no live animation).

    python run_sim.py                          # mode 1: insulin only, default patient, 24 h
    python run_sim.py --mode 2                 # mode 2: insulin + glucagon pumps
    python run_sim.py --patient adult#003 --hours 48
    python run_sim.py --show                   # also open the figure at the end

Mode 2 extras (glucagon pump and rescue controller):
    python run_sim.py --mode 2 --bolus-mult 2 --gl-max-flow 50 --gl-lag 5

Outputs go to results/ (git-ignored): <patient>.png and <patient>.csv
(mode 2 adds a _dual suffix).
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from carb_reporting import CarbReporting

import matplotlib
import matplotlib.pyplot as plt

from simglucose.actuator.pump import InsulinPump, PumpConfig, PumpModel
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import GLUCAGON_MODELS, T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.dual_env import DualAction, DualHormoneEnv
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario

RESULTS_DIR = Path(__file__).parent / "results"
LOW, HIGH = 70, 180  # target range, mg/dL

# (hour of day, grams of carbohydrate)
DEFAULT_MEALS = [(7, 45), (12, 70), (18, 80)]


# --------------------------------------------------------------------------
# Mode 1: insulin only (original single-pump setup)
# --------------------------------------------------------------------------
def build_env(patient_name, meals, seed):
    """Build the T1DSimEnv Object given paramaters"""
    start = datetime(2024, 1, 1, 0, 0)
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=start, scenario=meals)
    return T1DSimEnv(patient, sensor, pump, scenario), start


def run(patient_name, hours, meals, seed):
    """Run the closed loop and return history (df)"""
    env, start = build_env(patient_name, meals, seed)
    controller = BBController()
    obs, reward, done, info = env.reset()
    while env.time < start + timedelta(hours=hours):
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)
        if done:
            print("Stopped early: glucose left the safe simulation range.")
            break
    return env.show_history()


def summarize(df):
    """df --> dict"""
    bg = df["BG"]
    return {
        "mean": bg.mean(),
        "TIR": 100 * ((bg >= LOW) & (bg <= HIGH)).mean(),
        "TBR": 100 * (bg < LOW).mean(),
        "TAR": 100 * (bg > HIGH).mean(),
        "min": bg.min(),
        "max": bg.max(),
    }


# --------------------------------------------------------------------------
# Plot pieces (shared by both modes)
# --------------------------------------------------------------------------
def _title(patient_name, stats, extra=""):
    return (
        f"{patient_name}  |  TIR {stats['TIR']:.0f}%  "
        f"TBR {stats['TBR']:.0f}%  TAR {stats['TAR']:.0f}%  "
        f"(range {stats['min']:.0f}-{stats['max']:.0f}){extra}"
    )


def _plot_glucose(ax, df, title):
    """True BG vs what the sensor reports, with the target band"""
    ax.axhspan(LOW, HIGH, color="tab:green", alpha=0.12, label="Target range")
    ax.plot(df.index, df["BG"], color="black", lw=1.8, label="True BG")
    ax.plot(df.index, df["CGM"], color="tab:blue", lw=1, alpha=0.8, label="CGM")
    ax.axhline(LOW, color="tab:red", lw=0.8, ls="--")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_title(title)


def _plot_insulin(ax, t, ins_u_per_h):
    """Symlog axis so the small basal rate and the big boluses both show"""
    ax.step(t, ins_u_per_h, where="post", color="tab:purple", lw=1.2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Insulin (U/h)")


def _plot_meals(ax_cho, ax_bg, t, dt, carbs_g):
    """One bar per sample, label each meal with its total grams"""
    ax_cho.bar(t, carbs_g, width=dt / 1440, align="edge", color="tab:orange")
    eating = carbs_g > 0
    start = None
    for i, e in enumerate(list(eating) + [False]):
        if e and start is None:
            start = i
        elif not e and start is not None:
            ax_cho.annotate(f"{carbs_g[start:i].sum():.0f} g", (t[start], carbs_g[start:i].max()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
            ax_bg.axvline(t[start], color="tab:orange", lw=0.8, ls=":")
            start = None
    ax_cho.set_ylim(0, max(carbs_g.max() * 1.3, 1))
    ax_cho.set_ylabel("Carbs (g / sample)")
    ax_cho.set_xlabel("Time")


def _series(df):
    """History columns are one row shorter than the time index (no action at t=0)"""
    t = df.index[1:]
    dt = (df.index[1] - df.index[0]).total_seconds() / 60  # minutes per sample
    ins_u_per_h = df["insulin"].iloc[:-1].fillna(0).values * 60
    carbs_g = df["CHO"].iloc[:-1].fillna(0).values * dt  # grams per sample
    return t, dt, ins_u_per_h, carbs_g


def plot(df, patient_name, stats):
    """Basline plot function - stores in results generally"""
    fig, (ax_bg, ax_ins, ax_cho) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1]},
    )
    t, dt, ins_u_per_h, carbs_g = _series(df)

    _plot_glucose(ax_bg, df, _title(patient_name, stats))
    _plot_insulin(ax_ins, t, ins_u_per_h)
    _plot_meals(ax_cho, ax_bg, t, dt, carbs_g)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_dual(df, patient_name, stats, rescues):
    """Insulin + glucagon figure: glucose, insulin, glucagon, carbs"""
    fig, (ax_bg, ax_ins, ax_gl, ax_cho) = plt.subplots(
        4, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1]},
    )
    t, dt, ins_u_per_h, carbs_g = _series(df)
    gl_ug_per_min = df["glucagon"].iloc[:-1].fillna(0).values
    gl_cmd = df["glucagon_cmd"].iloc[:-1].fillna(0).values

    _plot_glucose(ax_bg, df, _title(patient_name, stats, f"  |  {rescues} glucagon rescue(s)"))
    _plot_insulin(ax_ins, t, ins_u_per_h)

    # Glucagon: commanded vs what the pump actually delivered
    ax_gl.step(t, gl_cmd, where="post", color="tab:gray", lw=1, ls="--", label="Commanded")
    ax_gl.step(t, gl_ug_per_min, where="post", color="tab:red", lw=1.5, label="Delivered")
    ax_gl.set_ylabel("Glucagon (ug/min)")
    ax_gl.set_ylim(bottom=0)
    ax_gl.legend(loc="upper right", fontsize=8)
    # mark the start of each glucagon dose on the glucose panel
    dosing = gl_ug_per_min > 0
    for i in range(len(t)):
        if dosing[i] and (i == 0 or not dosing[i - 1]):
            ax_bg.axvline(t[i], color="tab:red", lw=1, ls=":")

    _plot_meals(ax_cho, ax_bg, t, dt, carbs_g)
    if "CHO_reported" in df:
        reported_g = df["CHO_reported"].iloc[:-1].fillna(0).values * dt
        ax_cho.step(t, reported_g, where="post", color="tab:blue", lw=1.5,
                    label="Reported carbs")
        from matplotlib.patches import Patch
        handles, labels = ax_cho.get_legend_handles_labels()
        ax_cho.legend([Patch(color="tab:orange", label="Actual carbs"), *handles],
                      ["Actual carbs", *labels], loc="upper right", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Mode 2: insulin + glucagon
# --------------------------------------------------------------------------
class RescueController:
    """Basal-bolus insulin, plus a glucagon dose when CGM drops below a threshold.

    bolus_mult > 1 over-delivers insulin on purpose, to cause lows to rescue.
    """

    def __init__(self, bolus_mult=1.0, threshold=75.0, dose_ug=150.0, cooldown_min=90):
        self.bb = BBController()
        self.bolus_mult = bolus_mult
        self.threshold = threshold
        self.dose_ug = dose_ug
        self.cooldown = timedelta(minutes=cooldown_min)
        self.reset()

    def reset(self):
        self.bb.reset()
        self._last_dose = None
        self.rescues = 0

    def policy(self, observation, reward, done, **info):
        a = self.bb.policy(observation, reward, done, **info)
        glucagon = 0.0
        now = info["time"]
        low = observation.CGM < self.threshold
        if low and (self._last_dose is None or now - self._last_dose >= self.cooldown):
            # the action is a rate held for one sample, so spread the dose over it
            glucagon = self.dose_ug / info["sample_time"]
            self._last_dose = now
            self.rescues += 1
        return DualAction(a.basal, a.bolus * self.bolus_mult, glucagon)


def build_dual_env(patient_name, meals, seed, gl_config, glucagon_model=None):
    start = datetime(2024, 1, 1, 0, 0)
    patient = T1DPatient.withName(patient_name, glucagon_model=glucagon_model)  # None = library default
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    insulin_pump = InsulinPump.withName("Insulet")
    glucagon_pump = PumpModel(gl_config)
    scenario = CustomScenario(start_time=start, scenario=meals)
    return DualHormoneEnv(patient, sensor, insulin_pump, glucagon_pump, scenario), start


def run_dual(patient_name, hours, meals, seed, gl_config, controller, *, glucagon_model=None, reporting=None):
    """Run the two-pump closed loop; returns history (df)"""
    env, start = build_dual_env(patient_name, meals, seed, gl_config, glucagon_model)
    controller.reset()
    reporting = reporting or CarbReporting()
    obs, reward, done, info = env.reset()
    reported_meal = 0.0
    while env.time < start + timedelta(hours=hours):
        # Keep the original scenario and physiological meal input unchanged.
        # Only the synthetic meal announcement in the controller input differs.
        controller_info = dict(info, meal=reported_meal)
        action = controller.policy(obs, reward, done, **controller_info)
        obs, reward, done, info = env.step(action)
        reported_meal = reporting.report(info["meal"])
        if done:
            print("Stopped early: glucose left the safe simulation range.")
            break
    df = env.show_history()
    # CHO is recorded at the action row, not the next observation row.
    df["CHO_reported"] = [x[0] for x in reporting.history] + [float("nan")]
    df["carb_reporting_factor"] = [x[1] for x in reporting.history] + [float("nan")]
    return df


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def _save_and_report(df, fig, stem, stats, args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / f"{stem}.csv")
    png = output / f"{stem}.png"
    fig.savefig(png, dpi=150)
    print(f"Saved {png}")
    print({k: round(float(v), 1) for k, v in stats.items()})
    if args.show:
        plt.show()


def basic_run_and_plot(args):
    """Mode 1: insulin only"""
    meals = [(float(h), g) for h, g in DEFAULT_MEALS]

    df = run(args.patient, args.hours, meals, args.seed)
    stats = summarize(df)

    stem = args.patient.replace("#", "_")
    fig = plot(df, args.patient, stats)
    _save_and_report(df, fig, stem, stats, args)


def dual_run_and_plot(args):
    """Mode 2: insulin + glucagon pumps with a low-glucose rescue controller"""
    meals = [(float(h), g) for h, g in DEFAULT_MEALS]
    gl_config = PumpConfig(
        max_flow=args.gl_max_flow,
        lag_tau=args.gl_lag,
        delay=args.gl_delay,
        noise_std=args.gl_noise,
        reservoir=args.gl_reservoir,
        seed=args.seed,
    )
    controller = RescueController(
        bolus_mult=args.bolus_mult,
        threshold=args.rescue_threshold,
        dose_ug=args.rescue_dose,
    )

    reporting = CarbReporting(args.carb_factor, args.random_carb_reporting, args.seed)
    df = run_dual(args.patient, args.hours, meals, args.seed, gl_config, controller,
                  glucagon_model=args.glucagon_model, reporting=reporting)
    stats = summarize(df)
    stats["glucagon_ug"] = df["glucagon"].sum() * (df.index[1] - df.index[0]).total_seconds() / 60
    stats["rescues"] = controller.rescues

    stem = args.patient.replace("#", "_") + "_dual"
    fig = plot_dual(df, args.patient, stats, controller.rescues)
    _save_and_report(df, fig, stem, stats, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", type=int, choices=[1, 2], default=1,
                    help="1 = insulin only (original), 2 = insulin + glucagon")
    ap.add_argument("--patient", default="adolescent#001")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--show", action="store_true",
                    help="open the figure at the end (default: only save it)")

    g = ap.add_argument_group("mode 2 (glucagon)")
    g.add_argument("--carb-factor", type=float, default=1.0,
                   help="fraction of actual meal reported to simulated controller")
    g.add_argument("--random-carb-reporting", action="store_true",
                   help="draw a reproducible uniform reporting factor for each meal")
    g.add_argument("--bolus-mult", type=float, default=1.0,
                   help="scale meal boluses; >1 over-delivers insulin to cause lows (default 1)")
    g.add_argument("--rescue-threshold", type=float, default=75.0,
                   help="CGM below this triggers glucagon, mg/dL")
    g.add_argument("--rescue-dose", type=float, default=150.0, help="glucagon per rescue, ug")
    g.add_argument("--gl-max-flow", type=float, default=100.0,
                   help="glucagon pump max flow, ug/min")
    g.add_argument("--gl-lag", type=float, default=0.0, help="glucagon pump lag time constant, min")
    g.add_argument("--gl-delay", type=float, default=0.0, help="glucagon pump delay, min")
    g.add_argument("--gl-noise", type=float, default=0.0,
                   help="glucagon pump flow noise, fraction (0.05 = 5%%)")
    g.add_argument("--gl-reservoir", type=float, default=float("inf"),
                   help="glucagon reservoir size, ug")
    g.add_argument("--glucagon-model", choices=GLUCAGON_MODELS, default=None,
                   help="glucagon action model (default: the library default, legacy_linear)")
    args = ap.parse_args()
    if args.mode == 1 and (args.carb_factor != 1 or args.random_carb_reporting):
        ap.error("Carbohydrate reporting options require mode 2")
    if not 0 <= args.carb_factor <= 1:
        ap.error("--carb-factor must be between 0 and 1")

    if not args.show:
        matplotlib.use("Agg")  # no window, no animation

    if args.mode == 1:
        basic_run_and_plot(args)
    else:
        dual_run_and_plot(args)


if __name__ == "__main__":
    main()
