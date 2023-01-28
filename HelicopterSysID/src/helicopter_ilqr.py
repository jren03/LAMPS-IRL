import ray
import numpy as np
import copy

from linearized_helicopter_dynamics import linearized_heli_dynamics_2
from hover import DT, Q, R, Qfinal
from lqr import lqr_linearized_tv
from controller import LinearControllerWithFeedForwardAndNoOffset, Controller, LineSearchController
from helicopter import HelicopterModel

NUM_ILQR_ITERATIONS = 100
NUM_LINE_SEARCH_ITERATIONS = 20


@ray.remote
def linearize_dynamics_and_quadraticize_cost(
    state, next_state, control, target_state, target_control, params
):
    A_t, B_t = linearized_heli_dynamics_2(
        state,
        next_state,
        control,
        DT,
        params,
        offset=False,
    )
    C_x_t = Q[:12, :12] @ (state - target_state)
    C_u_t = R @ (control - target_control)
    C_xx_t = Q[:12, :12]
    C_uu_t = R.copy()
    return A_t, B_t, C_x_t, C_xx_t, C_u_t, C_uu_t


def solve_controller(
    x_result: np.ndarray, u_result: np.ndarray, target_states: np.ndarray,
    target_controls: np.ndarray, model: HelicopterModel, horizon: int
):
    result = ray.get(
        [
            linearize_dynamics_and_quadraticize_cost.remote(
                x_result[:, t], x_result[:, t+1], u_result[:, t],
                target_states[t, :], target_controls[t, :],
                model.params,
            )
            for t in range(horizon)
        ]
    )
    A, B, C_x, C_xx, C_u, C_uu = list(zip(*result))
    C_x_f = Qfinal[:12, :12] @ (x_result[:, horizon] -
                                target_states[horizon, :])
    C_xx_f = Qfinal[:12, :12]

    k, K = lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
    new_controller = LinearControllerWithFeedForwardAndNoOffset(
        k, K, x_result.T, u_result.T, time_invariant=False)

    return new_controller


def ilqr(model: HelicopterModel, horizon: int, initial_controller: Controller, rollout_fn, target_states: np.ndarray, target_controls: np.ndarray):
    controller = copy.deepcopy(initial_controller)
    x_result, u_result, cost = rollout_fn(controller, alpha=1.0)
    num_lqr_calls = 0
    # print("ILQR", cost)

    for _ in range(NUM_ILQR_ITERATIONS):

        new_controller = solve_controller(
            x_result, u_result, target_states, target_controls, model, horizon)
        num_lqr_calls += 1

        alpha_found = False
        alpha = 1.0
        for _ in range(NUM_LINE_SEARCH_ITERATIONS):
            new_x_result, new_u_result, new_cost = rollout_fn(
                new_controller, alpha)
            if new_cost < cost:
                controller = LineSearchController(new_controller, alpha)
                x_result = new_x_result.copy()
                u_result = new_u_result.copy()
                cost = new_cost
                alpha_found = True
                break
            alpha = 0.5 * alpha
        if not alpha_found:
            break

        # print("ILQR", cost, alpha)

    return controller, num_lqr_calls
