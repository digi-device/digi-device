import numpy as np
import pytest

from simglucose.actuator.pump import InsulinPump, PumpConfig, PumpModel


def test_default_is_passthrough():
    p = PumpModel()
    assert p.deliver(1.5) == 1.5
    assert p.deliver(0.0) == 0.0


def test_limits_resolution_deadband():
    p = PumpModel(PumpConfig(max_flow=2.0, resolution=0.1, deadband=0.05))
    assert p.deliver(0.04) == 0.0
    assert p.deliver(0.26) == pytest.approx(0.3)
    assert p.deliver(9.0) == 2.0


def test_delay():
    p = PumpModel(PumpConfig(delay=3))
    out = [p.deliver(1.0) for _ in range(5)]
    assert out == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_lag_step_response():
    tau = 5.0
    p = PumpModel(PumpConfig(lag_tau=tau))
    out = [p.deliver(1.0) for _ in range(20)]
    assert out[4] == pytest.approx(1 - np.exp(-5 / tau))
    assert out[-1] < 1.0 and out == sorted(out)


def test_gain_bias_noise_repeatable():
    cfg = PumpConfig(gain=0.9, bias=0.01, noise_std=0.05, seed=1)
    a = [PumpModel(cfg).deliver(1.0) for _ in range(1)]
    p1, p2 = PumpModel(cfg), PumpModel(cfg)
    assert [p1.deliver(1.0) for _ in range(5)] == [p2.deliver(1.0) for _ in range(5)]
    assert abs(a[0] - 0.91) < 0.2


def test_reservoir_runs_out():
    p = PumpModel(PumpConfig(reservoir=2.5))
    out = [p.deliver(1.0) for _ in range(4)]
    assert out == [1.0, 1.0, 0.5, 0.0]
    assert p.empty
    p.reset()
    assert p.reservoir_remaining == 2.5


def test_insulin_pump_unchanged():
    p = InsulinPump.withName('Cozmo')
    assert p.basal(0.0123) == pytest.approx(0.0125 * 0 + round(0.0123 * 6000 / 0.05) * 0.05 / 6000)
    assert p.bolus(100) == 75.0
    assert p.basal(-1) == 0.0
    p.reset()
    assert isinstance(p, PumpModel)
