import numpy as np


class Controller:
    def __init__(self):
        pass

    def act(self, x, t, alpha=None):
        pass


class LinearController(Controller):
    def __init__(self, K, P, nominal_state, nominal_control, time_invariant=False):
        self.K = K
        self.P = P
        self.nominal_state = nominal_state
        self.nominal_control = nominal_control
        self.time_invariant = time_invariant

    def act(self, x, t, alpha=None):
        nominal_control = (
            self.nominal_control if self.time_invariant else self.nominal_control[t]
        )
        nominal_state = (
            self.nominal_state if self.time_invariant else self.nominal_state[t]
        )
        K = self.K if self.time_invariant else self.K[t]

        # return nominal_control + K @ (np.append(x - nominal_state, 1))
        return nominal_control + K @ (x - nominal_state)


class LinearControllerWithFeedForwardAndNoOffset(Controller):
    def __init__(self, k, K, nominal_state, nominal_control, time_invariant=False):
        self.k = k
        self.K = K
        self.nominal_state = nominal_state
        self.nominal_control = nominal_control
        self.time_invariant = time_invariant

    def act(self, x, t, alpha=1.0):
        nominal_control = (
            self.nominal_control if self.time_invariant else self.nominal_control[t]
        )
        nominal_state = (
            self.nominal_state if self.time_invariant else self.nominal_state[t]
        )
        K = self.K if self.time_invariant else self.K[t]
        k = self.k if self.time_invariant else self.k[t]

        alpha = 1.0 if alpha is None else alpha
        return (
            (1 - alpha) * nominal_control
            + alpha * k
            + K @ (x - (1 - alpha) * nominal_state)
        )


class LineSearchController(Controller):
    def __init__(self, controller, alpha):
        self.controller = controller
        self.alpha = alpha

    def act(self, x, t, alpha=None):
        return self.controller.act(x, t, self.alpha)
