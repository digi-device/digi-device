from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simglucose.actuator.pump import InsulinPump, PumpConfig, PumpModel
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient import t1dpatient
from simglucose.patient.t1dpatient import Action, T1DPatient, load_glucagon_params
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.dual_env import DualAction, DualHormoneEnv
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario

# Gsub at t=500 and t=719 from the ORIGINAL (insulin-only) patient model, with
# basal insulin, a 5 U bolus and 4 x 20 g CHO starting at t=400.
ORIGINAL_REFERENCE = {
    "adolescent#001": (164.6615, 125.7314),
    "adult#007": (150.6342, 81.489),
    "child#004": (110.4819, 17.7022),
}


@pytest.mark.parametrize("name", ORIGINAL_REFERENCE)
def test_zero_glucagon_matches_original_model(name):
    # exact: the glucagon states are solved apart from the glucose ODE
    p = T1DPatient.withName(name)
    basal = p._params.u2ss * p._params.BW / 6000
    got = {}
    for t in range(720):
        meal = 20 if t in (400, 401, 402, 403) else 0
        p.step(Action(CHO=meal, insulin=basal + (5 if t == 400 else 0)))
        if t in (500, 719):
            got[t] = p.observation.Gsub
    assert got[500] == pytest.approx(ORIGINAL_REFERENCE[name][0], abs=1e-4)  # references are rounded to 4 decimals
    assert got[719] == pytest.approx(ORIGINAL_REFERENCE[name][1], abs=1e-4)  # references are rounded to 4 decimals


def _fasting_run(name, dose, minutes=240):
    p = T1DPatient.withName(name)
    basal = p._params.u2ss * p._params.BW / 6000
    bg = []
    for t in range(minutes):
        p.step(Action(CHO=0, insulin=basal, glucagon=dose if t == 0 else 0))
        bg.append(p.observation.Gsub)
    return np.array(bg)


def test_basal_only_stays_flat():
    bg = _fasting_run("adult#001", 0)
    assert bg.max() - bg.min() < 0.5


def test_glucagon_raises_bg_with_delay_and_dose_response():
    base = _fasting_run("adult#001", 0)
    small = _fasting_run("adult#001", 150) - base
    big = _fasting_run("adult#001", 1000) - base
    assert 10 < small.max() < big.max()
    assert big.max() > 80
    assert 20 <= big.argmax() <= 90          # peaks within ~1.5 h
    assert big[5] < 0.2 * big.max()          # not instantaneous
    assert abs(big[-1]) < big.max()          # effect fades


def _env(env_cls, **kw):
    start = datetime(2024, 1, 1)
    patient = T1DPatient.withName("adolescent#001")
    sensor = CGMSensor.withName("Dexcom", seed=1)
    scenario = CustomScenario(start, [(1.0, 45)])
    pump = InsulinPump.withName("Insulet")
    if env_cls is DualHormoneEnv:
        return DualHormoneEnv(patient, sensor, pump, kw["gl_pump"], scenario), start
    return T1DSimEnv(patient, sensor, pump, scenario), start


def _run(env, start, hours, glucagon=0.0, gl_from=None):
    ctrl = BBController()
    obs, reward, done, info = env.reset()
    while env.time < start + timedelta(hours=hours):
        a = ctrl.policy(obs, reward, done, **info)
        if glucagon and gl_from is not None and env.time >= start + timedelta(hours=gl_from):
            a = DualAction(a.basal, a.bolus, glucagon)
        obs, reward, done, info = env.step(a)
    return env.show_history()


def test_dual_env_matches_single_pump_env_without_glucagon():
    e1, s = _env(T1DSimEnv)
    e2, _ = _env(DualHormoneEnv, gl_pump=PumpModel())
    h1, h2 = _run(e1, s, 6), _run(e2, s, 6)
    np.testing.assert_allclose(h2["BG"], h1["BG"], atol=0.05)
    np.testing.assert_allclose(h2["insulin"].iloc[:-1], h1["insulin"].iloc[:-1], atol=1e-4)
    assert (h2["glucagon"].dropna() == 0).all()


def test_dual_env_glucagon_raises_bg_and_logs_pump_state():
    cfg = PumpConfig(max_flow=50.0, reservoir=300.0)
    e, s = _env(DualHormoneEnv, gl_pump=PumpModel(cfg))
    # ask for 100 ug/min for 10 min (1000 ug) but only 300 ug are in the reservoir
    start_hr = 3.0
    ctrl = BBController()
    obs, reward, done, info = e.reset()
    while e.time < s + timedelta(hours=6):
        a = ctrl.policy(obs, reward, done, **info)
        if s + timedelta(hours=start_hr) <= e.time < s + timedelta(hours=start_hr, minutes=10):
            a = DualAction(a.basal, a.bolus, 100.0)
        obs, reward, done, info = e.step(a)
    h = e.show_history()
    assert h["glucagon_cmd"].max() == pytest.approx(100.0)
    assert h["glucagon"].max() <= 50.0 + 1e-9                  # flow limit
    assert h["glucagon"].sum() * e.sample_time == pytest.approx(300.0, rel=1e-3)  # reservoir
    assert h["glucagon_reservoir"].dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)

    # and it moved glucose compared with no glucagon
    e0, _ = _env(DualHormoneEnv, gl_pump=PumpModel())
    h0 = _run(e0, s, 6)
    assert h["BG"].max() > h0["BG"].max() + 5


# --------------------------------------------------------------------------
# Glucagon model switch: the original model stays available as 'legacy_linear'
# --------------------------------------------------------------------------
# The legacy row, written out so these tests keep checking the old model even after the
# default row in glucagon_params.csv moves to a newer one.
LEGACY_ROW = pd.Series({"Name": "default", "kgl_abs": 0.1, "kgl_cl": 0.12, "Vgl": 0.2,
                        "kgl_eff": 0.1, "kgn": 1.1, "model": "legacy_linear"})
REPO = Path(__file__).resolve().parents[2]
BASELINE_CURVES = REPO / "baselines" / "glucagon_v0_linear" / "dose_curves.csv"


def test_default_row_selects_legacy_linear():
    assert load_glucagon_params()["model"] == "legacy_linear"
    assert T1DPatient.withName("adult#001")._gmodel == "legacy_linear"


def test_row_without_model_value_is_legacy_linear():
    row = LEGACY_ROW.drop("model")
    assert T1DPatient.withName("adult#001", glucagon_params=row)._gmodel == "legacy_linear"


def test_unknown_glucagon_model_fails_early():
    row = LEGACY_ROW.copy()
    row["model"] = "no_such_model"
    with pytest.raises(ValueError, match="unknown glucagon model"):
        T1DPatient.withName("adult#001", glucagon_params=row)


def test_explicit_legacy_row_gives_identical_response():
    base = _fasting_run("adult#001", 0)
    default = _fasting_run("adult#001", 300) - base

    p = T1DPatient.withName("adult#001", glucagon_params=LEGACY_ROW)
    basal = p._params.u2ss * p._params.BW / 6000
    bg = []
    for t in range(240):
        p.step(Action(CHO=0, insulin=basal, glucagon=300 if t == 0 else 0))
        bg.append(p.observation.Gsub)
    np.testing.assert_allclose(np.array(bg) - base, default, atol=1e-9)


@pytest.mark.skipif(not BASELINE_CURVES.exists(), reason="baselines/glucagon_v0_linear is not present")
def test_legacy_linear_reproduces_committed_baseline(monkeypatch):
    # Re-runs the exact protocol behind baselines/glucagon_v0_linear and compares curves.
    # If this fails after a model change, the legacy branch changed behaviour.
    monkeypatch.setattr(t1dpatient, "load_glucagon_params", lambda name=None, model=None: LEGACY_ROW.copy())
    monkeypatch.syspath_prepend(str(REPO))
    import report_graphs  # imports from the repo root; also sets a non-interactive matplotlib backend

    new = report_graphs.run_hypo_protocol("adolescent#001", 200, 1.0, seed=1).sort_values("t_min")
    old = pd.read_csv(BASELINE_CURVES)
    old = old[(old.patient == "adolescent#001") & (old.dose_ug == 200)].sort_values("t_min")
    assert len(new) == len(old)
    for col in ("BG", "g0", "g1", "g2", "egp_gl"):
        np.testing.assert_allclose(new[col].to_numpy(), old[col].to_numpy(), rtol=1e-6, atol=1e-6,
                                   equal_nan=True, err_msg=col)


# --------------------------------------------------------------------------
# linear_wendt_pk: Wendt 2017 pharmacokinetics, same linear effect
# --------------------------------------------------------------------------
def _glucagon_run(p, rates):
    """Step a fasting patient at basal insulin; rates[t] is glucagon in ug/min. Returns (BG, plasma pg/ml)."""
    basal = p._params.u2ss * p._params.BW / 6000
    bg, conc = [], []
    for u in rates:
        p.step(Action(CHO=0, insulin=basal, glucagon=u))
        bg.append(p.observation.Gsub)
        conc.append(p.plasma_glucagon_pg_ml)
    return np.array(bg), np.array(conc)


def test_params_are_selected_by_model():
    legacy = load_glucagon_params()
    wendt = load_glucagon_params(model="linear_wendt_pk")
    assert legacy["model"] == "legacy_linear" and legacy["kgl_abs"] == 0.1
    # means of Wendt 2017 Table 3 over the 8 patients
    assert wendt["kgl_abs"] == pytest.approx(0.338 / 8)
    assert wendt["kgl_cl"] == pytest.approx(1.368 / 8)
    assert wendt["Cl_FC"] == pytest.approx(1025 / 8)
    assert wendt["Cb"] == pytest.approx(85.0 / 8)
    with pytest.raises(ValueError, match="unknown glucagon model"):
        load_glucagon_params(model="no_such_model")


def test_legacy_has_no_plasma_concentration():
    assert np.isnan(T1DPatient.withName("adult#001").plasma_glucagon_pg_ml)


def test_wendt_pk_basal_and_steady_state_concentration():
    p = T1DPatient.withName("adult#001", glucagon_model="linear_wendt_pk")
    g = load_glucagon_params(model="linear_wendt_pk")
    assert p.plasma_glucagon_pg_ml == pytest.approx(g["Cb"])
    # constant infusion u: at steady state, C - Cb = u / (BW * clearance), here in pg/ml
    u = 1.0  # ug/min
    _, conc = _glucagon_run(p, [u] * 600)
    assert conc[-1] - g["Cb"] == pytest.approx(1e6 * u / (p._params.BW * g["Cl_FC"]), rel=1e-6)


def test_wendt_pk_plasma_peak_time_matches_clinical_range():
    p = T1DPatient.withName("adult#001", glucagon_model="linear_wendt_pk")
    _, conc = _glucagon_run(p, [200.0] + [0.0] * 119)  # 200 ug in the first minute
    t_peak = np.argmax(conc) + 1  # conc[i] is at the end of minute i+1
    assert 7.5 <= t_peak <= 19.1  # range of Wendt 2017 Table 3


def test_wendt_pk_zero_glucagon_matches_original_model():
    name = "adolescent#001"
    p = T1DPatient.withName(name, glucagon_model="linear_wendt_pk")
    basal = p._params.u2ss * p._params.BW / 6000
    got = {}
    for t in range(720):
        meal = 20 if t in (400, 401, 402, 403) else 0
        p.step(Action(CHO=meal, insulin=basal + (5 if t == 400 else 0)))
        if t in (500, 719):
            got[t] = p.observation.Gsub
    assert got[500] == pytest.approx(ORIGINAL_REFERENCE[name][0], abs=1e-4)
    assert got[719] == pytest.approx(ORIGINAL_REFERENCE[name][1], abs=1e-4)


def test_wendt_pk_glucagon_still_raises_bg():
    rates = [200.0] + [0.0] * 239
    base, _ = _glucagon_run(T1DPatient.withName("adult#001", glucagon_model="linear_wendt_pk"), [0.0] * 240)
    dosed, _ = _glucagon_run(T1DPatient.withName("adult#001", glucagon_model="linear_wendt_pk"), rates)
    assert (dosed - base).max() > 1.0
