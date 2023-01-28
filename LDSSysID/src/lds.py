import numpy as np


class LDS(object):
    def __init__(
        self,
        initial_state: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        horizon: int,
        time_varying: bool,
    ):
        self.initial_state = initial_state.copy()
        self.A = A.copy()
        self.B = B.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.Qf = Qf.copy()
        self.horizon = horizon
        self.time_varying = time_varying

    def step(self, x: np.ndarray, u: np.ndarray, t: int, noise: np.ndarray = None):
        xnext = (
            self.A @ x + self.B @ u
            if not self.time_varying
            else self.A[t] @ x + self.B[t] @ u
        )
        return xnext if noise is None else xnext + noise

    def reset(self):
        return self.initial_state

    @property
    def params(self):
        if self.time_varying:
            raise ("Not defined for time varying models")
        return np.concatenate([self.A.flatten(), self.B.flatten()])

    def unpack_params(self, params: np.ndarray):
        if self.time_varying:
            raise ("Not defined for time varying models")
        return params[: self.A.size].reshape(self.A.shape), params[
            self.A.size :
        ].reshape(self.B.shape)

    @property
    def nominal_state(self):
        return np.zeros(self.Q.shape[0])

    @property
    def nominal_control(self):
        return np.zeros(self.R.shape[0])
