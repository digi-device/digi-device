from datetime import datetime, timedelta

import numpy as np
import pytest

from simglucose.actuator.pump import InsulinPump, PumpConfig, PumpModel
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import Action, T1DPatient
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
