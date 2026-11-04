import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pkg_resources

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')
logger = logging.getLogger(__name__)


@dataclass
class PumpConfig:
    """Tunable pump parameters. Flow is in units/min, reservoir in units.

    Defaults describe an ideal pump (no lag, no noise, unlimited reservoir),
    so a default PumpModel passes commands straight through.
    """
    min_flow: float = 0.0            # smallest nonzero flow it can deliver
    max_flow: float = float('inf')   # flow ceiling
    resolution: float = 0.0          # flow step size (0 = continuous)
    deadband: float = 0.0            # commands <= this deliver nothing
    lag_tau: float = 0.0             # first-order lag time constant, min (0 = none)
    delay: float = 0.0               # pure transport delay, min
    gain: float = 1.0                # multiplicative calibration error
    bias: float = 0.0                # additive flow error
    noise_std: float = 0.0           # random error, fraction of flow (1 sigma)
    reservoir: float = float('inf')  # total units available
    seed: Optional[int] = None       # noise seed, for repeatable runs


class PumpModel(object):
    """Generic pump: command in, delivered flow out.

    deliver(command, dt) applies, in order: deadband, quantization, flow
    limits, transport delay, first-order lag, gain/bias, noise, reservoir.
    Use one instance per hormone (insulin, glucagon, bucket acid, ...).
    """

    def __init__(self, config=None):
        self.config = config if config is not None else PumpConfig()
        self.reset()

    def reset(self):
        self._rng = np.random.default_rng(self.config.seed)
        self._queue = deque()
        self._lagged = 0.0
        self.reservoir_remaining = self.config.reservoir
        self.total_delivered = 0.0

    @property
    def empty(self):
        return self.reservoir_remaining <= 0.0

    def condition(self, command):
        """Command -> what the pump firmware would accept (no dynamics)."""
        c = self.config
        if command <= max(c.deadband, 0.0):
            return 0.0
        if c.resolution > 0:
            command = np.round(command / c.resolution) * c.resolution
        return float(min(max(command, c.min_flow), c.max_flow))

    def deliver(self, command, dt=1.0):
        """Advance the pump by dt minutes; return delivered flow (units/min)."""
        c = self.config
        cmd = self.condition(command)

        # transport delay (tubing / cannula)
        n = int(round(c.delay / dt))
        self._queue.append(cmd)
        cmd = self._queue.popleft() if len(self._queue) > n else 0.0

        # first-order lag (exact discretization, stable for any dt)
        if c.lag_tau > 0:
            alpha = 1.0 - np.exp(-dt / c.lag_tau)
            self._lagged += alpha * (cmd - self._lagged)
            flow = self._lagged
        else:
            flow = cmd

        # calibration error and noise only matter when pumping
        if flow > 0:
            flow = flow * c.gain + c.bias
            if c.noise_std > 0:
                flow *= 1.0 + c.noise_std * self._rng.standard_normal()
        flow = max(flow, 0.0)

        # reservoir: can't deliver more than what is left
        flow = min(flow, self.reservoir_remaining / dt)
        self.reservoir_remaining -= flow * dt
        self.total_delivered += flow * dt
        return flow


class InsulinPump(PumpModel):
    """Existing simglucose pump (Cozmo, Insulet) built on PumpModel.

    basal() and bolus() behave exactly as before. The extra dynamics from
    PumpModel (deliver, lag, noise, reservoir) are available but off by
    default.
    """
    U2PMOL = 6000

    def __init__(self, params, config=None):
        self._params = params
        if config is None:
            config = PumpConfig(
                max_flow=params['max_basal'] + params['max_bolus'],
                resolution=params['inc_basal'] / self.U2PMOL,
            )
        super(InsulinPump, self).__init__(config)

    @classmethod
    def withName(cls, name, config=None):
        pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
        params = pump_params.loc[pump_params.Name == name].squeeze()
        return cls(params, config)

    def bolus(self, amount):
        bol = amount * self.U2PMOL  # convert from U/min to pmol/min
        bol = np.round(bol / self._params['inc_bolus']
                       ) * self._params['inc_bolus']
        bol = bol / self.U2PMOL     # convert from pmol/min to U/min
        bol = min(bol, self._params['max_bolus'])
        bol = max(bol, self._params['min_bolus'])
        return bol

    def basal(self, amount):
        bas = amount * self.U2PMOL  # convert from U/min to pmol/min
        bas = np.round(bas / self._params['inc_basal']
                       ) * self._params['inc_basal']
        bas = bas / self.U2PMOL     # convert from pmol/min to U/min
        bas = min(bas, self._params['max_basal'])
        bas = max(bas, self._params['min_basal'])
        return bas

    def reset(self):
        logger.info('Resetting insulin pump ...')
        super(InsulinPump, self).reset()
