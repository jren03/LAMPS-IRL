import numpy as np
from collections import deque
import argparse
import copy

from widetree import WideTree, simulate_episode

DEFAULT_NUM_ITERATIONS = 100
DATASET_SIZE = 10000

def generate_model_class(n_leaves: int, num_models: int):
    assert num_models > 1, "More than one model in model class"
    model_class = []
    # Add bad model
    model_class.append(WideTree(n_leaves, "bad"))
    # Add several good models
    for _ in range(num_models-1):
        model_class.append(WideTree(n_leaves, "good"))

    return model_class

def compute_model_advantage_loss(dataset: deque, values: np.ndarray, model: WideTree):
    loss = 0.0
    for transition in dataset:
        predicted_state, _ = model.step(transition[0], transition[1])
        loss += np.abs(values[transition[3] - 1] - values[predicted_state - 1])

    return loss / len(dataset)


def compute_classification_loss(dataset: deque, model: WideTree):
    loss = 0.0
    for transition in dataset:
        predicted_state, _ = model.step(transition[0], transition[1])
        loss += 1 - float(predicted_state == transition[3])

    return loss / len(dataset)

def run(real_world: WideTree, n_leaves: int, num_models: int, moment_based: bool):
    # Generate model class
    model_class = generate_model_class(n_leaves, num_models)
    # Hedge weights
    weights = (1.0/num_models) * np.ones(num_models)
    all_weights = [weights.copy()]
    # Hedge epsilon
    eps = 0.9
    # Choose an initial model
    idx = np.random.randint(num_models)
    model = model_class[idx]
    # Compute policy and values
    policy, values = model.optimal_policy_and_values()
    # Initialize losses
    all_losses = []
    losses = [0.0 for _ in range(num_models)]
    all_losses.append(copy.deepcopy(losses))
    # Initialize dataset
    dataset = deque(maxlen=DATASET_SIZE)
    # Get expert controller
    expert_policy, _ = real_world.optimal_policy_and_values()

    # Start iterations
    for i in range(DEFAULT_NUM_ITERATIONS):
        # Rollout using current policy in real world
        transitions, total_return = simulate_episode(real_world, policy)
        dataset += transitions
        print("Cost to go", total_return)
        # Rollout using expert policy in real world
        transitions, total_return = simulate_episode(real_world, expert_policy)
        dataset += transitions

        # Update model
        for j in range(num_models):
            losses[j] += compute_model_advantage_loss(dataset, values, model_class[j]) if moment_based else compute_classification_loss(dataset, model_class[j])

        # Update hedge weights
        for j in range(num_models):
            loss = compute_model_advantage_loss(dataset, values, model_class[j]) if moment_based else compute_classification_loss(dataset, model_class[j])
            weights[j] = weights[j] * np.exp(-eps * loss)
        all_weights.append(weights / np.sum(weights))
        
        # Find best model
        # Do FTL
        # model = model_class[np.argmin(losses)]
        # print(losses)
        all_losses.append(copy.deepcopy(losses))
        # print("Selected", np.argmin(losses))

        # Do Hedge
        probs = weights / np.sum(weights)
        model_idx = np.random.choice(np.arange(num_models), p=probs)
        model = model_class[model_idx]
        print(weights / np.sum(weights))
        print("Selected", model_idx)

        # Compute policy and values
        policy, values = model.optimal_policy_and_values()

    return np.array(all_losses), np.array(all_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_leaves", type=int, default=100)
    parser.add_argument("--n_models", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    real_world = WideTree(args.n_leaves, "real")
    # MLE
    np.random.seed(args.seed)
    mle_losses, mle_weights = run(real_world, args.n_leaves, args.n_models, moment_based=False)
    # MOMENT BASED
    np.random.seed(args.seed)
    moment_based_losses, moment_based_weights = run(real_world, args.n_leaves, args.n_models, moment_based=True)

    np.save(f"data/mle_{args.seed}.npy", mle_weights)
    np.save(f"data/moment_based_{args.seed}.npy", moment_based_weights)