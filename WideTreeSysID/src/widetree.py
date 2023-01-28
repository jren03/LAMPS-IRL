import numpy as np

SUCCESS_PROB = 0.7

class WideTree:
    def __init__(self, n_leaves: int, model: str):
        assert n_leaves >= 4, "Should at least have 4 leaf nodes"
        assert n_leaves % 4 == 0, "Number of leaves should be a multiple of 4"
        assert model in ["good", "bad", "real"], "model should be one of [good, bad, real]"

        self.bucket_size = n_leaves//4
        self.n_leaves = n_leaves
        self.model = model
        if model == "real":
            self.construct_real_model()
        elif model == "good":
            self.construct_good_model()
        else:
            self.construct_bad_model()

    def construct_real_model(self):
        self.left = 2
        self.right = 3
        self.first_parent = 4
        self.second_parent = 5
        self.first_bucket = np.arange(6, 6+self.bucket_size)
        self.second_bucket = np.arange(6 + self.bucket_size, 6 + 2*self.bucket_size)
        self.third_bucket = np.arange(6 + 2*self.bucket_size, 6 + 3*self.bucket_size)
        self.fourth_bucket = np.arange(6 + 3*self.bucket_size, 6 + 4*self.bucket_size)

    def construct_good_model(self):
        self.left = 2
        self.right = 3
        self.first_parent = 5
        self.second_parent = 4
        # Generate a random permutation of leaves
        leaves = np.random.permutation(np.arange(6, 6 + self.n_leaves))
        self.first_bucket = leaves[0:self.bucket_size]
        self.second_bucket = leaves[self.bucket_size:2*self.bucket_size]
        self.third_bucket = leaves[2*self.bucket_size:3*self.bucket_size]
        self.fourth_bucket = leaves[3*self.bucket_size:4*self.bucket_size]

    def construct_bad_model(self):
        # Switch dynamics at the root
        self.left = 3
        self.right = 2
        self.first_parent = 4
        self.second_parent = 5
        self.first_bucket = np.arange(6, 6+self.bucket_size)
        self.second_bucket = np.arange(6 + self.bucket_size, 6 + 2*self.bucket_size)
        self.third_bucket = np.arange(6 + 2*self.bucket_size, 6 + 3*self.bucket_size)
        self.fourth_bucket = np.arange(6 + 3*self.bucket_size, 6 + 4*self.bucket_size)

    def reset(self):
        return 1

    def num_nodes(self):
        return 5 + self.n_leaves

    def num_non_leaves(self):
        return 5

    def step(self, state: int, action: int):
        assert state >= 1 and state <= self.num_nodes()
        assert action in [0, 1]

        if state == 1:
            if action == 0:
                return (self.left, 0) if np.random.rand() < SUCCESS_PROB else (self.right, 0)
            else:
                return (self.right, 0) if np.random.rand() < SUCCESS_PROB else (self.left, 0)
        elif state == 2:
            return self.first_parent, 1
        elif state == 3:
            return self.second_parent, 0
        elif state == self.first_parent:
            if action == 0:
                return np.random.choice(self.first_bucket), 0
            else:
                return np.random.choice(self.second_bucket), 0
        elif state == self.second_parent:
            if action == 0:
                return np.random.choice(self.third_bucket), 0
            else:
                return np.random.choice(self.fourth_bucket), 0

        raise("Should never reach here")

    def random_policy_and_values(self):
        policy = np.random.choice([0, 1], self.num_nodes())
        values = np.zeros(self.num_nodes())
        if policy[0] == 0 and self.left == 2:
            values[0] = 1 * SUCCESS_PROB
        elif policy[0] == 1 and self.right == 2:
            values[0] = 1 * SUCCESS_PROB
        values[1] = 1
        return policy, values

    def optimal_policy_and_values(self):
        policy = np.random.choice([0, 1], self.num_nodes())
        if self.left == 2:
            policy[0] = 1  # Take right
        else:
            policy[0] = 0  # Take left
        values = np.zeros(self.num_nodes())
        values[0] = (1 - SUCCESS_PROB)
        values[1] = 1
        return policy, values

    def check_terminal(self, state: int):
        return state > self.num_non_leaves()


def simulate_episode(widetree: WideTree, policy: np.ndarray):
    state = widetree.reset()
    transitions = []
    total_return = 0
    while not widetree.check_terminal(state):
        action = policy[state-1]
        next_state, cost = widetree.step(state, action)
        transitions.append((state, action, cost, next_state))
        total_return += cost
        state = next_state

    return transitions, total_return