"""Closed-loop environment with two pumps: insulin and glucagon."""
from collections import namedtuple

import pandas as pd

from simglucose.patient.t1dpatient import Action
from simglucose.simulation.env import T1DSimEnv

# glucagon is a rate in ug/min. Optional, so insulin-only controllers still work.
DualAction = namedtuple("dual_ctrller_action", ["basal", "bolus", "glucagon"],
                        defaults=(0.0,))


class DualHormoneEnv(T1DSimEnv):
    """T1DSimEnv with an extra glucagon pump.

    Insulin goes through insulin_pump.basal()/bolus() as before, then
    insulin_pump.deliver() (lag, delay, noise, reservoir). Glucagon goes
    through glucagon_pump.deliver(). With default (ideal) PumpModel settings
    and glucagon = 0, behaviour matches T1DSimEnv.
    """

    def __init__(self, patient, sensor, insulin_pump, glucagon_pump, scenario):
        self.glucagon_pump = glucagon_pump
        super(DualHormoneEnv, self).__init__(patient, sensor, insulin_pump, scenario)

    @property
    def insulin_pump(self):
        return self.pump

    def _reset(self):
        super(DualHormoneEnv, self)._reset()
        self.glucagon_hist = []
        self.insulin_cmd_hist = []
        self.glucagon_cmd_hist = []
        self.insulin_reservoir_hist = []
        self.glucagon_reservoir_hist = []
        self._acc = dict(glucagon=0.0, insulin_cmd=0.0, glucagon_cmd=0.0)

    def reset(self):
        self.glucagon_pump.reset()
        return super(DualHormoneEnv, self).reset()

    def mini_step(self, action):
        dt = self.patient.sample_time  # min
        patient_action = self.scenario.get_action(self.time)
        glucagon_cmd = max(getattr(action, "glucagon", 0.0), 0.0)
        insulin_cmd = (self.pump.basal(action.basal)
                       + self.pump.bolus(action.bolus))

        insulin = self.pump.deliver(insulin_cmd, dt)
        glucagon = self.glucagon_pump.deliver(glucagon_cmd, dt)

        self.patient.step(Action(insulin=insulin, CHO=patient_action.meal,
                                 glucagon=glucagon))

        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        self._acc["glucagon"] += glucagon
        self._acc["insulin_cmd"] += insulin_cmd
        self._acc["glucagon_cmd"] += glucagon_cmd
        return patient_action.meal, insulin, BG, CGM

    def step(self, action, reward_fun=None):
        self._acc = dict(glucagon=0.0, insulin_cmd=0.0, glucagon_cmd=0.0)
        if reward_fun is None:
            result = super(DualHormoneEnv, self).step(action)
        else:
            result = super(DualHormoneEnv, self).step(action, reward_fun)

        n = int(self.sample_time)
        self.glucagon_hist.append(self._acc["glucagon"] / n)
        self.insulin_cmd_hist.append(self._acc["insulin_cmd"] / n)
        self.glucagon_cmd_hist.append(self._acc["glucagon_cmd"] / n)
        self.insulin_reservoir_hist.append(self.pump.reservoir_remaining)
        self.glucagon_reservoir_hist.append(self.glucagon_pump.reservoir_remaining)

        info = result.info if hasattr(result, "info") else result[3]
        info["glucagon"] = self.glucagon_hist[-1]
        return result

    def show_history(self):
        df = super(DualHormoneEnv, self).show_history()
        # same convention as insulin: one entry per action, so one shorter
        # than the time index
        for col, hist in [
            ("glucagon", self.glucagon_hist),
            ("insulin_cmd", self.insulin_cmd_hist),
            ("glucagon_cmd", self.glucagon_cmd_hist),
            ("insulin_reservoir", self.insulin_reservoir_hist),
            ("glucagon_reservoir", self.glucagon_reservoir_hist),
        ]:
            df[col] = pd.Series(hist, dtype=float).set_axis(df.index[: len(hist)])
        return df
