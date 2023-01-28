import numpy as np
import copy
from collections import deque
import ray
import argparse
import matplotlib.pyplot as plt

from model_fit import initial_model, fit_model, loss_dataset
from hover_controller import optimal_hover_controller_ilqr, optimal_hover_controller_psdp_2
from helicopter import HelicopterModel
from hover import rollout_hover_controller, HOVER_AT_ZERO, HOVER_TRIMS
from controller import Controller

DATASET_SIZE = 10000
HORIZON = 100
NOISE_CONTROL_EXPERT_CONTROLLER = 0.0001
BASELINE_PROB = 0.5
DEFAULT_NUM_ITERATIONS = 20
DEFAULT_NUM_SAMPLES_PER_ITERATION = 100
EVAL_DATASET_NUM_ROLLOUTS = 10
EVAL_NUM_ROLLOUTS = 100


def evaluate_controller(controller: Controller, real_world: HelicopterModel, add_noise: bool):
    cost = 0.0
    for _ in range(EVAL_NUM_ROLLOUTS):
        cost += rollout_hover_controller(controller, real_world,
                                         HORIZON, early_stop=False, add_noise=add_noise)[-1]
    cost /= EVAL_NUM_ROLLOUTS
    return cost


def evaluate_statistics(controller: Controller, expert_controller: Controller, real_world: HelicopterModel, model: HelicopterModel, eval_dataset: deque, add_noise: bool):
    # Evaluate controller
    cost = evaluate_controller(controller, real_world, add_noise)
    print("Cost in real world", cost)
    # Evaluate controller in model
    cost_model = rollout_hover_controller(
        controller, model, HORIZON, early_stop=False, add_noise=False)[-1]
    #print("Cost in model", costs_model[-1])
    # Evaluate expert controller in model
    expert_cost_model = rollout_hover_controller(
        expert_controller, model, HORIZON, early_stop=False, add_noise=False)[-1]
    #print("Expert cost in model", expert_cost_model[-1])
    # Evaluate model error
    model_error = loss_dataset(eval_dataset, model)
    #print("Model error", model_errors[-1])

    return cost, cost_model, expert_cost_model, model_error


def hover_sys_id(pdl: bool, add_noise: bool = True):
    nominal_model = initial_model()
    model = copy.deepcopy(nominal_model)
    controller, num_lqr_calls = optimal_hover_controller_ilqr(model, HORIZON)
    real_world = HelicopterModel()

    dataset = deque(maxlen=DATASET_SIZE)
    expert_controller, _ = optimal_hover_controller_ilqr(real_world, HORIZON)

    eval_dataset = deque(maxlen=EVAL_DATASET_NUM_ROLLOUTS * HORIZON)
    for _ in range(EVAL_DATASET_NUM_ROLLOUTS):
        x_expert, u_expert, cost_expert = rollout_hover_controller(
            expert_controller, real_world, HORIZON, early_stop=False, add_noise=add_noise)
        # print(cost_expert)
        for t in range(HORIZON):
            eval_dataset.append(
                (x_expert[:, t], u_expert[:, t], x_expert[:, t+1]))

    costs, lqr_calls, model_errors, costs_model, expert_costs_model = [], [], [], [], []

    lqr_calls.append(num_lqr_calls)
    cost, cost_model, expert_cost_model, model_error = evaluate_statistics(
        controller, expert_controller, real_world, model, eval_dataset, add_noise)
    costs.append(cost)
    costs_model.append(cost_model)
    expert_costs_model.append(expert_cost_model)
    model_errors.append(model_error)

    for n in range(DEFAULT_NUM_ITERATIONS):
        print("Iteration", n)

        # Rollout controller in real world
        x_controller, u_controller, _ = rollout_hover_controller(
            controller, real_world, HORIZON, early_stop=True, add_noise=add_noise)

        # # Rollout expert controller in real world
        # x_expert, u_expert, _ = rollout_hover_controller(
        #     expert_controller, real_world, HORIZON, early_stop=True, add_noise=add_noise)  # NOTE: Changed to no early stop

        for k in range(DEFAULT_NUM_SAMPLES_PER_ITERATION):
            toss = np.random.rand()
            if toss < BASELINE_PROB or u_controller.shape[1] == 0:
                # t = np.random.randint(u_expert.shape[1])
                idx = np.random.randint(len(eval_dataset))
                # state, control, next_state = x_expert[:,
                #                                       t], u_expert[:, t], x_expert[:, t+1]
                state, control, next_state = eval_dataset[idx]
            else:
                t = np.random.randint(u_controller.shape[1])
                state, control, next_state = x_controller[:,
                                                          t], u_controller[:, t], x_controller[:, t+1]

            # Add to dataset
            dataset.append((state, control, next_state))

        # Fit new model
        model = fit_model(dataset, nominal_model)

        # Compute new controller
        result = optimal_hover_controller_psdp_2(
            model, HORIZON) if pdl else optimal_hover_controller_ilqr(model, HORIZON)
        controller = copy.deepcopy(result[0])
        num_lqr_calls += result[1]

        lqr_calls.append(num_lqr_calls)
        cost, cost_model, expert_cost_model, model_error = evaluate_statistics(
            controller, expert_controller, real_world, model, eval_dataset, add_noise)
        costs.append(cost)
        costs_model.append(cost_model)
        expert_costs_model.append(expert_cost_model)
        model_errors.append(model_error)

    # Rollout expert controller in real world
    best_cost = evaluate_controller(expert_controller, real_world, add_noise)
    print("Cost of optimal trajectory", best_cost)

    return controller, costs, best_cost, costs_model, expert_costs_model, model_errors, lqr_calls, model


if __name__ == '__main__':
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    ag_controller, ag_costs, best_cost, ag_costs_model, ag_expert_cost_model, ag_model_errors, ag_lqr_calls, _ = hover_sys_id(
        pdl=False, add_noise=True)

    np.random.seed(args.seed)
    pdl_controller, pdl_costs, best_cost, pdl_costs_model, pdl_expert_cost_model, pdl_model_errors, pdl_lqr_calls, _ = hover_sys_id(
        pdl=True, add_noise=True)

    np.save(f"data/ag_costs_{args.seed}.npy", ag_costs)
    np.save(f"data/ag_costs_model_{args.seed}.npy", ag_costs_model)
    np.save(f"data/ag_expert_cost_model_{args.seed}.npy", ag_expert_cost_model)
    np.save(f"data/ag_model_errors_{args.seed}.npy", ag_model_errors)
    np.save(f"data/ag_lqr_calls_{args.seed}.npy", ag_lqr_calls)

    np.save(f"data/pdl_costs_{args.seed}.npy", pdl_costs)
    np.save(f"data/pdl_costs_model_{args.seed}.npy", pdl_costs_model)
    np.save(
        f"data/pdl_expert_cost_model_{args.seed}.npy", pdl_expert_cost_model)
    np.save(f"data/pdl_model_errors_{args.seed}.npy", pdl_model_errors)
    np.save(f"data/pdl_lqr_calls_{args.seed}.npy", pdl_lqr_calls)

    np.save(f"data/best_cost_{args.seed}.npy", [best_cost])
