import numpy as np
import copy
from collections import deque
from scipy.optimize import minimize

from helicopter import ParameterizedHelicopterModel, HelicopterModel, step_batch_fn
from hover import DT

NUM_RANDOM_RESTARTS = 4

def initial_model():
    true_model = HelicopterModel()
    params = true_model.params + \
        np.random.rand(len(true_model.params))

    return ParameterizedHelicopterModel(params)


def construct_training_data(dataset: deque):
    states, controls, next_states = tuple(zip(*dataset))
    states, controls, next_states = np.array(
        states), np.array(controls), np.array(next_states)
    return states, controls, next_states


def loss_dataset(dataset: deque, model: HelicopterModel):
    states, controls, next_states = construct_training_data(dataset)
    return loss(states, controls, next_states, model.params)


def loss(states: np.ndarray, controls: np.ndarray, next_states: np.ndarray, params: np.ndarray):
    predicted_next_states = step_batch_fn(states, controls, DT, params)
    return np.mean(np.linalg.norm(predicted_next_states - next_states, axis=1) ** 2)


def fit_model(dataset, nominal_model):
    states, controls, next_states = construct_training_data(dataset)

    def loss_fn(params):
        return loss(states, controls, next_states, params)

    result = minimize(
        loss_fn,
        nominal_model.params,
        method="BFGS",
        options={
            "disp": False,
            "gtol": 1e-3,  # NOTE: Changed from 1e-5
        },
    )

    params = result.x.copy()
    obj = copy.deepcopy(result.fun)

    # Run optimization from random starts to ensure global optimization
    for _ in range(NUM_RANDOM_RESTARTS):
        result = minimize(
            loss_fn,
            nominal_model.random_params(),
            method="BFGS",
            options={
                "disp": False,
                "gtol": 1e-3,
            }
        )
        if result.fun < obj:
            params = result.x.copy()
            obj = copy.deepcopy(result.fun)

    return ParameterizedHelicopterModel(params)
