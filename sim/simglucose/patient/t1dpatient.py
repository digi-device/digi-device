from .base import Patient
import numpy as np
from scipy.integrate import ode
from scipy.linalg import expm
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources

logger = logging.getLogger(__name__)

# glucagon is in ug/min (subcutaneous infusion rate); defaults to 0 so
# insulin-only code keeps working unchanged
Action = namedtuple("patient_action", ["CHO", "insulin", "glucagon"], defaults=(0.0,))
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)
GLUCAGON_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/glucagon_params.csv"
)
N_BASE_STATES = 13  # original UVA/Padova states (solved by the ODE solver)
N_STATES = 16  # + 3 glucagon states (x13, x14, x15), solved exactly, see below


# Glucagon-action models, selected by the 'model' column of glucagon_params.csv.
#   legacy_linear:   glucagon effect is added to EGP in proportion to the delayed plasma
#                    glucagon (kgn * g2). Kept unchanged so earlier results stay reproducible.
#   linear_wendt_pk: same linear effect, but the pharmacokinetics use the fitted constants of
#                    Wendt et al. 2017 (Table 3 means): kgl_abs = k1, kgl_cl = k2, plasma
#                    clearance Cl_FC (ml/kg/min) and basal level Cb (pg/ml). The effect chain is
#                    driven by plasma glucagon above basal instead of g1/Vgl. Vgl is unused.
GLUCAGON_MODELS = ("legacy_linear", "linear_wendt_pk")
DEFAULT_GLUCAGON_MODEL = "legacy_linear"


def load_glucagon_params(name=None, model=None):
    """Glucagon parameters for a patient and model; falls back to the model's 'default' row.

    model=None selects DEFAULT_GLUCAGON_MODEL.
    """
    model = DEFAULT_GLUCAGON_MODEL if model is None else model
    if model not in GLUCAGON_MODELS:
        raise ValueError(f"unknown glucagon model '{model}'; choose one of {GLUCAGON_MODELS}")
    table = pd.read_csv(GLUCAGON_PARA_FILE)
    table = table.loc[table.model.fillna(DEFAULT_GLUCAGON_MODEL) == model]
    row = table.loc[table.Name == name]
    if row.empty:
        row = table.loc[table.Name == "default"]
    if row.empty:
        raise ValueError(f"no glucagon parameters for model '{model}' in {GLUCAGON_PARA_FILE}")
    return row.squeeze()


def glucagon_model_name(gparams):
    """Which glucagon model a parameter row selects. A row with no 'model' value is legacy_linear."""
    if gparams is None:
        return DEFAULT_GLUCAGON_MODEL
    name = getattr(gparams, "model", None)
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return DEFAULT_GLUCAGON_MODEL
    if name not in GLUCAGON_MODELS:
        raise ValueError(f"unknown glucagon model '{name}'; choose one of {GLUCAGON_MODELS}")
    return name


def glucagon_egp(gparams, glu_effect):
    """Glucagon's contribution to endogenous glucose production (mg/kg/min).

    glu_effect is the delayed plasma-glucagon signal from the glucagon states. New models branch here.
    """
    if gparams is None:
        return 0.0
    model = glucagon_model_name(gparams)
    if model in ("legacy_linear", "linear_wendt_pk"):
        return gparams.kgn * glu_effect
    raise NotImplementedError(model)  # listed in GLUCAGON_MODELS but not implemented above


def glucagon_effect_coupling(gparams):
    """Gain from the plasma state g1 (ug/kg) to the input of the delayed-effect state g2 (ug/L).

    legacy_linear divides by a volume (Vgl). linear_wendt_pk uses the apparent clearance: plasma
    glucagon above basal is kgl_cl * g1 / Cl_FC (ug/ml), times 1000 for ug/L.
    """
    if glucagon_model_name(gparams) == "linear_wendt_pk":
        return gparams.kgl_eff * 1000.0 * gparams.kgl_cl / gparams.Cl_FC
    return gparams.kgl_eff / gparams.Vgl


def glucagon_plasma_pg_ml(gparams, g1):
    """Plasma glucagon in pg/ml for a plasma state g1 (ug/kg): Cb + 1e6 * kgl_cl * g1 / Cl_FC.

    Same as Wendt et al. C = k2*Z2/(W*Cl_FC) + Cb. NaN for legacy_linear, which has no clearance or basal level.
    """
    if glucagon_model_name(gparams) == "legacy_linear":
        return float("nan")
    return float(gparams.Cb + 1e6 * gparams.kgl_cl * g1 / gparams.Cl_FC)


class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(
        self,
        params,
        init_state=None,
        random_init_bg=False,
        seed=None,
        t0=0,
        glucagon_params=None,
        glucagon_model=None,
    ):
        """
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
            - glucagon_params: pandas sequence from glucagon_params.csv. If
              not given, the 'default' row (or this patient's row) is used.
            - glucagon_model: which glucagon model's rows to use (see
              GLUCAGON_MODELS). Ignored if glucagon_params is given.
        """
        self._params = params
        if glucagon_params is None:
            glucagon_params = load_glucagon_params(params.Name, glucagon_model)
        self._gparams = glucagon_params
        self._gmodel = glucagon_model_name(glucagon_params)  # fails early on an unknown model
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        """13 UVA/Padova states followed by 3 glucagon states."""
        return np.concatenate([self._odesolver.y, self._gstate])

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    @property
    def plasma_glucagon_pg_ml(self):
        """Plasma glucagon (pg/ml), or NaN for models without a clearance (legacy_linear)."""
        return glucagon_plasma_pg_ml(self._gparams, self._gstate[1])

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # Glucagon states depend only on the glucagon input (never on glucose),
        # so they are advanced exactly, separately from the glucose ODE. This
        # keeps the original 13-state ODE and its solver behaviour untouched.
        g_start = self._gstate[2]
        self._gstate = self._Ad @ self._gstate + self._Bd * (
            action.glucagon / self._params.BW  # ug/min -> ug/kg/min
        )
        # glucagon action on the liver, averaged over the step
        glu_effect = 0.5 * (g_start + self._gstate[2])

        # ODE solver
        self._odesolver.set_f_params(
            action,
            self._params,
            self._last_Qsto,
            self._last_foodtaken,
            self._gparams,
            glu_effect,
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken, gparams=None, glu_effect=0.0):
        dxdt = np.zeros(len(x))
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
        # details.
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucagon raises production. glu_effect is the (delayed) plasma
        # glucagon above basal, so with no glucagon given this term is 0.
        # The model used is chosen by the 'model' column of glucagon_params.csv.
        EGPt = EGPt + glucagon_egp(gparams, glu_effect)
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))

        return dxdt

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _glucagon_discrete_system(self):
        """Exact one-step (SAMPLE_TIME) update of the linear glucagon chain.

        States: g[0] subcutaneous depot (ug/kg), g[1] plasma glucagon above
        basal (ug/kg), g[2] delayed action on the liver (ug/L). Input is the
        infusion rate in ug/kg/min, held constant over the step.
            dg0 = u - kgl_abs*g0
            dg1 = kgl_abs*g0 - kgl_cl*g1
            dg2 = -kgl_eff*(g2 - g1/Vgl)          (legacy_linear)
            dg2 = -kgl_eff*(g2 - 1000*kgl_cl*g1/Cl_FC)   (linear_wendt_pk: plasma glucagon above basal)
        This follows the same depot -> plasma -> delayed-action chain as insulin.
        """
        gp = self._gparams
        A = np.array(
            [
                [-gp.kgl_abs, 0.0, 0.0],
                [gp.kgl_abs, -gp.kgl_cl, 0.0],
                [0.0, glucagon_effect_coupling(gp), -gp.kgl_eff],
            ]
        )
        B = np.array([1.0, 0.0, 0.0])
        Ad = expm(A * self.SAMPLE_TIME)
        Bd = np.linalg.solve(A, (Ad - np.eye(3)) @ B)
        return Ad, Bd

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        """
        Reset the patient state to default intial state
        """
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:15].values)
        else:
            self.init_state = np.copy(self._init_state)
        # glucagon states start at zero (no glucagon above basal)
        if len(self.init_state) == N_STATES:
            self._gstate = np.array(self.init_state[N_BASE_STATES:], dtype=float)
            self.init_state = self.init_state[:N_BASE_STATES]
        else:
            self._gstate = np.zeros(N_STATES - N_BASE_STATES)
        self._Ad, self._Bd = self._glucagon_discrete_system()

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            # Only randomize glucose related states, x4, x5, and x13
            mean = [
                1.0 * self.init_state[3],
                1.0 * self.init_state[4],
                1.0 * self.init_state[12],
            ]
            cov = np.diag(
                [
                    0.1 * self.init_state[3],
                    0.1 * self.init_state[4],
                    0.1 * self.init_state[12],
                ]
            )
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0, glucagon=0)
        self.is_eating = False
        self.planned_meal = 0


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    p = T1DPatient.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()
