import numpy as np
import ray

from linearized_helicopter_dynamics import linearized_heli_dynamics_2
from hover import HOVER_AT_ZERO, HOVER_TRIMS, DT, Q, R, rollout_hover_controller, Qfinal
from lqr import lqr_lti
from controller import LinearController, Controller
from helicopter_ilqr import ilqr, solve_controller
from helicopter import HelicopterModel


def optimal_hover_controller_psdp(model: HelicopterModel):
    A, B = linearized_heli_dynamics_2(
        HOVER_AT_ZERO, HOVER_AT_ZERO, HOVER_TRIMS, DT, model.params)

    K, P = lqr_lti(A, B, Q, R)
    num_lqr_calls = 1
    return LinearController(K, P, HOVER_AT_ZERO, HOVER_TRIMS, time_invariant=True), num_lqr_calls


def optimal_hover_controller_psdp_2(model: HelicopterModel, horizon: int):
    num_lqr_calls = 1
    return solve_controller(
        np.array([HOVER_AT_ZERO.copy() for _ in range(horizon + 1)]).T,
        np.array([HOVER_TRIMS.copy() for _ in range(horizon)]).T,
        np.array([HOVER_AT_ZERO.copy() for _ in range(horizon + 1)]),
        np.array([HOVER_TRIMS.copy() for _ in range(horizon)]),
        model,
        horizon
    ), num_lqr_calls


def optimal_hover_controller_ilqr(model: HelicopterModel, horizon: int):
    def rollout_fn(controller: Controller, alpha: float):
        return rollout_hover_controller(
            controller, model, horizon, early_stop=False, alpha=alpha, add_noise=False)

    target_states = np.array([HOVER_AT_ZERO.copy()
                             for _ in range(horizon + 1)])
    target_controls = np.array([HOVER_TRIMS.copy() for _ in range(horizon)])

    initial_controller, num_lqr_calls = optimal_hover_controller_psdp_2(
        model, horizon)

    result = ilqr(model, horizon, initial_controller,
                  rollout_fn, target_states, target_controls)
    num_lqr_calls += result[1]

    return result[0], num_lqr_calls
