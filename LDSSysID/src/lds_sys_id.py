import numpy as np
import argparse
from lds import LDS
from cost import stage_cost, terminal_cost
from controller import Controller
from lds_controller import optimal_controller_lds
from collections import deque
from fit_model import fit_model_mle, fit_model_moment

import matplotlib.pyplot as plt
import copy
import ray

DATASET_SIZE = 10000
NOISE_VARIANCE = 0.0
DEFAULT_NUM_ITERATIONS = 50
DEFAULT_SAMPLES_PER_ITERATION = 100
BASELINE_PROB = 0.5
STATE_SIZE = 5
CONTROL_SIZE = 1
VALUE_FN_BUFFER_SIZE = 100
HORIZON = 100


def rollout(
    model: LDS,
    controller: Controller,
    alpha: float = None,
    add_noise: bool = False,
):
    xs = []
    us = []
    cost = 0.0

    state = model.reset()
    xs.append(state.copy())

    for t in range(model.horizon):
        control = controller.act(state, t, alpha=alpha)
        us.append(control.copy())
        next_state = model.step(
            state,
            control,
            t,
            noise=np.random.randn(state.shape[0]) * NOISE_VARIANCE
            if add_noise
            else None,
        )
        cost += stage_cost(state, control, model.Q, model.R)
        state = next_state.copy()
        xs.append(state.copy())

    cost += terminal_cost(state, model.Qf)

    return np.array(xs), np.array(us), cost


def construct_real_world(horizon: int) -> LDS:
    # A = np.array([[1.0, 1.0], [-3.0, 1.0]])
    # B = np.array([[1.0], [3.0]])
    # Q = 0.0001 * np.eye(STATE_SIZE, STATE_SIZE)
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    R = np.eye(CONTROL_SIZE, CONTROL_SIZE)
    Qf = np.eye(STATE_SIZE, STATE_SIZE)
    # Qf = np.zeros((STATE_SIZE, STATE_SIZE))
    x0 = 0.1 * np.ones(STATE_SIZE)

    A = np.eye(STATE_SIZE, STATE_SIZE)
    B = np.eye(STATE_SIZE, CONTROL_SIZE)

    As = [A.copy() for _ in range(horizon)]
    for i in range(horizon):
        # As[i][1, 0] = -5.0 if i % 2 == 0 else -1.0
        As[i] *= 0.5 if i % 2 == 0 else 1.5
    Bs = [B.copy() for _ in range(horizon)]

    return A, B, LDS(x0, As, Bs, Q, R, Qf, horizon, time_varying=True)


def construct_model(A: np.ndarray, B: np.ndarray, real_world: LDS) -> LDS:
    # eps = 1e-1
    # Ahat = A + eps * np.eye(STATE_SIZE, STATE_SIZE)
    # Bhat = B + eps * np.eye(STATE_SIZE, CONTROL_SIZE)
    Ahat = np.eye(STATE_SIZE, STATE_SIZE)
    Bhat = np.eye(STATE_SIZE, CONTROL_SIZE)

    return LDS(
        real_world.initial_state,
        Ahat,
        Bhat,
        real_world.Q,
        real_world.R,
        real_world.Qf,
        real_world.horizon,
        time_varying=False,
    )


def lds_sys_id(mle: bool):
    A, B, real_world = construct_real_world(HORIZON)
    model = construct_model(A, B, real_world)
    initial_model = construct_model(A, B, real_world)

    controller = optimal_controller_lds(model)
    dataset = deque(maxlen=DATASET_SIZE)
    expert_controller = optimal_controller_lds(real_world)

    costs = []
    costs.append(rollout(real_world, controller)[2])
    print("Cost in real world", costs[-1])

    Ps = deque(maxlen=VALUE_FN_BUFFER_SIZE)
    Ps.append(copy.deepcopy(controller.P))

    for n in range(DEFAULT_NUM_ITERATIONS):
        print("Iteration", n)

        # Rollout controller in real world
        x_controller, u_controller, _ = rollout(real_world, controller, add_noise=True)

        # Rollout expert controller in real world
        x_expert, u_expert, _ = rollout(real_world, expert_controller, add_noise=True)

        for k in range(DEFAULT_SAMPLES_PER_ITERATION):
            toss = np.random.rand()
            t = np.random.randint(HORIZON)
            if toss < BASELINE_PROB:
                state, control, next_state = (
                    x_expert[t, :],
                    u_expert[t, :],
                    x_expert[t + 1, :],
                )
                is_expert = True
            else:
                state, control, next_state = (
                    x_controller[t, :],
                    u_controller[t, :],
                    x_controller[t + 1, :],
                )
                is_expert = False

            # Add to dataset
            dataset.append((state, control, next_state, t, is_expert))

        # Fit new model
        model = (
            fit_model_mle(dataset, initial_model)
            if mle
            else fit_model_moment(dataset, Ps, initial_model)
        )

        # Compute new controller
        controller = optimal_controller_lds(model)
        Ps.append(copy.deepcopy(controller.P))

        costs.append(rollout(real_world, controller)[2])
        print("Cost in real world", costs[-1])

    # Rollout expert controller in real world
    best_cost = rollout(real_world, expert_controller)[2]
    print("Optimal cost in real world", best_cost)

    return costs, best_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    ray.init()
    costs, best_cost = lds_sys_id(mle=True)
    np.random.seed(args.seed)
    moment_costs, _ = lds_sys_id(mle=False)

    # np.save(f"data/mle_{args.seed}.npy", costs)
    # np.save(f"data/moment_based_{args.seed}.npy", moment_costs)
