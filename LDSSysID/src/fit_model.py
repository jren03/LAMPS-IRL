import numpy as np
from scipy.optimize import minimize
from collections import deque
from lds import LDS
from typing import List

from numba import njit
import ray

ALPHA = 0.01


def construct_training_data(dataset: deque):
    states, controls, next_states, timesteps, is_expert = tuple(zip(*dataset))
    states, controls, next_states, timesteps, is_expert = (
        np.array(states),
        np.array(controls),
        np.array(next_states),
        np.array(timesteps),
        np.array(is_expert),
    )
    return states, controls, next_states, timesteps, is_expert


def loss_mle(
    states: np.ndarray,
    controls: np.ndarray,
    next_states: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
):
    predicted_next_states = states @ A.T + controls @ B.T
    return np.mean(np.linalg.norm(predicted_next_states - next_states, axis=1) ** 2)


@njit
def compute_moment_loss(
    state: np.ndarray,
    control: np.ndarray,
    next_state: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    is_expert: bool,
):
    next_moment = next_state.T @ P @ next_state
    predicted_next_state = A @ state + B @ control
    predicted_next_moment = predicted_next_state.T @ P @ predicted_next_state

    # TODO: L1 or L2?
    # return np.abs(next_moment - predicted_next_moment)
    return (next_moment - predicted_next_moment) ** 2


@njit
def regularization(params: np.ndarray):
    return np.linalg.norm(params) ** 2


@ray.remote
def loss_moment(
    states: np.ndarray,
    controls: np.ndarray,
    next_states: np.ndarray,
    timesteps: np.ndarray,
    is_expert: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    P: List[np.ndarray],
):
    result = [
        compute_moment_loss(
            states[idx, :],
            controls[idx, :],
            next_states[idx, :],
            A,
            B,
            P[timesteps[idx] + 1],
            is_expert[idx],
        )
        for idx in range(timesteps.shape[0])
    ]

    return np.mean(result)


def fit_model_mle(dataset: deque, nominal_model: LDS):
    states, controls, next_states, _, _ = construct_training_data(dataset)

    def loss_fn(params):
        A, B = nominal_model.unpack_params(params)
        return loss_mle(states, controls, next_states, A, B)

    result = minimize(
        loss_fn,
        nominal_model.params,
        method="BFGS",
        options={
            "disp": False,
            "gtol": 1e-5,
        },
    )

    params = result.x.copy()
    A, B = nominal_model.unpack_params(params)
    return LDS(
        nominal_model.initial_state,
        A,
        B,
        nominal_model.Q,
        nominal_model.R,
        nominal_model.Qf,
        nominal_model.horizon,
        nominal_model.time_varying,
    )


def fit_model_moment(dataset: deque, Ps: List[List[np.ndarray]], nominal_model: LDS):
    states, controls, next_states, timesteps, is_expert = construct_training_data(
        dataset
    )
    (
        states_remote,
        controls_remote,
        next_states_remote,
        timesteps_remote,
        is_expert_remote,
    ) = (
        ray.put(states),
        ray.put(controls),
        ray.put(next_states),
        ray.put(timesteps),
        ray.put(is_expert),
    )
    Ps_remote = [ray.put(P) for P in Ps]

    def loss_fn(params):
        A, B = nominal_model.unpack_params(params)
        A_remote, B_remote = ray.put(A), ray.put(B)
        result = ray.get(
            [
                loss_moment.remote(
                    states_remote,
                    controls_remote,
                    next_states_remote,
                    timesteps_remote,
                    is_expert_remote,
                    A_remote,
                    B_remote,
                    P_remote,
                )
                for P_remote in Ps_remote
            ]
        )
        return np.mean(result) + ALPHA * regularization(params)

    result = minimize(
        loss_fn,
        nominal_model.params,
        method="BFGS",
        options={
            "disp": False,
            "gtol": 1e-5,
        },
    )

    params = result.x.copy()
    A, B = nominal_model.unpack_params(params)
    return LDS(
        nominal_model.initial_state,
        A,
        B,
        nominal_model.Q,
        nominal_model.R,
        nominal_model.Qf,
        nominal_model.horizon,
        nominal_model.time_varying,
    )
