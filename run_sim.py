"""Run one simulated patient and save a summary graph (no live animation).

    python run_sim.py                          # default patient, 24 h
    python run_sim.py --patient adult#003 --hours 48
    python run_sim.py --show                   # also open the figure at the end

Outputs go to results/ (git-ignored): <patient>.png and <patient>.csv
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as plt

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario

RESULTS_DIR = Path(__file__).parent / "results"
LOW, HIGH = 70, 180  # target range, mg/dL

# (hour of day, grams of carbohydrate)
DEFAULT_MEALS = [(7, 45), (12, 70), (18, 80)]


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


def plot(df, patient_name, stats):
    """Basline plot function - stores in results generally"""

    fig, (ax_bg, ax_ins, ax_cho) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1]},
    )

    # Glucose: true BG vs what the sensor reports, with the target band
    ax_bg.axhspan(LOW, HIGH, color="tab:green", alpha=0.12, label="Target range")
    ax_bg.plot(df.index, df["BG"], color="black", lw=1.8, label="True BG")
    ax_bg.plot(df.index, df["CGM"], color="tab:blue", lw=1, alpha=0.8, label="CGM")
    ax_bg.axhline(LOW, color="tab:red", lw=0.8, ls="--")
    ax_bg.set_ylabel("Glucose (mg/dL)")
    ax_bg.legend(loc="upper right", ncol=3, fontsize=8)
    ax_bg.set_title(
        f"{patient_name}  |  TIR {stats['TIR']:.0f}%  "
        f"TBR {stats['TBR']:.0f}%  TAR {stats['TAR']:.0f}%  "
        f"(range {stats['min']:.0f}-{stats['max']:.0f})"
    )

    # History columns are one row shorter than the time index (no action at t=0)
    t = df.index[1:]
    dt = (df.index[1] - df.index[0]).total_seconds() / 60  # minutes per sample
    ins_u_per_h = df["insulin"].iloc[:-1].fillna(0).values * 60
    carbs_g = df["CHO"].iloc[:-1].fillna(0).values * dt  # grams per sample

    # Insulin: symlog axis so the small basal rate and the big boluses both show
    ax_ins.step(t, ins_u_per_h, where="post", color="tab:purple", lw=1.2)
    ax_ins.set_yscale("symlog", linthresh=1)
    ax_ins.set_ylim(bottom=0)
    ax_ins.set_ylabel("Insulin (U/h)")

    # Meals: one bar per sample, label each meal with its total grams
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

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def basic_run_and_plot():
    """Main function for now"""

    # Setup args for simulator
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="adolescent#001")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", action="store_true",
                    help="open the figure at the end (default: only save it)")
    args = ap.parse_args()

    if not args.show:
        plt.use("Agg")  # no window, no animation

    # Setup patient env stuff
    meals = [(float(h), g) for h, g in DEFAULT_MEALS]

    # Run sim
    df = run(args.patient, args.hours, meals, args.seed)
    stats = summarize(df)

    # Generate plots and format data
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = args.patient.replace("#", "_")
    df.to_csv(RESULTS_DIR / f"{stem}.csv")
    fig = plot(df, args.patient, stats)
    png = RESULTS_DIR / f"{stem}.png"
    fig.savefig(png, dpi=150)

    print(f"Saved {png}")
    print({k: round(float(v), 1) for k, v in stats.items()})
    if args.show:
        plt.show()


if __name__ == "__main__":
    basic_run_and_plot()
