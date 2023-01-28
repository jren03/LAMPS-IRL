import numpy as np


def state_cost(x: np.ndarray, Q: np.ndarray):
    return 0.5 * x.T @ Q @ x


def control_cost(u: np.ndarray, R: np.ndarray):
    return 0.5 * u.T @ R @ u


def stage_cost(x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray):
    return state_cost(x, Q) + control_cost(u, R)


def terminal_cost(x: np.ndarray, Qf: np.ndarray):
    return state_cost(x, Qf)
