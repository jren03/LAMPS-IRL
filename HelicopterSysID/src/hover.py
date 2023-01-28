import numpy as np

from controller import Controller
from helicopter import HelicopterModel, step_fn
from cost import cost_state, cost_control, cost_final

HOVER_AT_ZERO = np.zeros(12)
HOVER_TRIMS = np.zeros(4)
Q = np.eye(13)
R = np.eye(4)
Qfinal = Q.copy()
DT = 0.05


def get_hover_noise():
    return np.random.randn(6) * 0.1


def rollout_hover_controller(controller: Controller, model: HelicopterModel, horizon: int, early_stop: bool = False, alpha: float = None, add_noise: bool = True):
    x_result = np.zeros((12, horizon + 1))
    x_result[:, 0] = HOVER_AT_ZERO.copy()

    u_result = np.zeros((4, horizon))
    cost = 0.0

    for t in range(horizon):
        u_result[:, t] = controller.act(x_result[:, t], t, alpha=alpha)

        cost += cost_state(x_result[:, t], HOVER_AT_ZERO, Q)
        cost += cost_control(u_result[:, t], HOVER_TRIMS, R)

        if (early_stop and np.linalg.norm(np.concatenate([x_result[:, t] - HOVER_AT_ZERO, u_result[:, t] - HOVER_TRIMS])) > 5):
            print("Stopping early at t:", t)
            return x_result[:, :t+1], u_result[:, :t], cost

        noise_F_t = get_hover_noise() if add_noise else np.zeros(6)

        x_result[:, t+1] = step_fn(x_result[:, t],
                                   u_result[:, t], DT, model.params, noise=noise_F_t)

    cost += cost_final(x_result[:, horizon], HOVER_AT_ZERO, Qfinal)

    # if early_stop:
    #     print("Reached end of horizon", horizon)

    return x_result, u_result, cost
