import numpy as np
from numba import njit

from axis_angle_dynamics_update import axis_angle_dynamics_update, axis_angle_dynamics_update_batch
from quaternion_from_axis_rotation import quaternion_from_axis_rotation, quaternion_from_axis_rotation_batch
from express_vector_in_quat_frame import express_vector_in_quat_frame, express_vector_in_quat_frame_batch
from rotate_vector import rotate_vector, rotate_vector_batch


NED_DOT_IDXS = np.arange(0, 3)
NED_IDXS = np.arange(3, 6)
PQR_IDXS = np.arange(6, 9)
AXIS_ANGLE_IDXS = np.arange(9, 12)
CONTROL_LIMITS = 1e2  # NOTE: Changed from 1e5


class HelicopterModel:
    def __init__(self):
        ## Mass and inertia
        self.m = 5  # kg
        self.Ixx = 0.3
        self.Iyy = 0.3
        self.Izz = 0.3
        self.Ixy = self.Ixz = self.Iyz = 0
        self.g = 9.81

        # Aerodynamic forces parameters
        self.Tx = np.array([0, -3.47, 13.20]) * self.Ixx
        self.Ty = np.array([0, -3.06, -9.21]) * self.Iyy
        self.Tz = np.array([0, -2.58, 14.84]) * self.Izz
        self.Fx = -0.048 * self.m
        self.Fy = np.array([0, -0.12]) * self.m
        self.Fz = np.array([-9.81, -0.0005, -27.5]) * self.m

    @property
    def params(self):
        return np.concatenate(
            [
                [self.m, self.Ixx, self.Iyy, self.Izz],
                self.Tx,
                self.Ty,
                self.Tz,
                [self.Fx],
                self.Fy,
                self.Fz,
                [self.g],
            ]
        )

    def random_params(self):
        N = self.params.shape[0]
        return self.params + 0.1 * np.random.randn(N)


class ParameterizedHelicopterModel(HelicopterModel):
    def __init__(self, params: np.ndarray):
        self.m = params[0]
        self.Ixx = params[1]
        self.Iyy = params[2]
        self.Izz = params[3]
        self.Ixy = self.Iyz = self.Ixz = 0.0
        self.Tx = params[4:7].copy()
        self.Ty = params[7:10].copy()
        self.Tz = params[10:13].copy()
        self.Fx = params[13]
        self.Fy = params[14:16].copy()
        self.Fz = params[16:19].copy()
        self.g = params[19]


# Dynamics

@njit
def step_fn(
    x0: np.ndarray,
    u0: np.ndarray,
    dt: float,
    params: np.ndarray,
    noise: np.ndarray = None,
):
    X0 = np.reshape(np.ascontiguousarray(x0), (1, -1))
    U0 = np.reshape(np.ascontiguousarray(u0), (1, -1))

    X1 = step_batch_fn(X0, U0, dt, params, noise=noise)

    return X1[0, :]


@njit
def step_batch_fn(X: np.ndarray, U: np.ndarray, dt: float, params: np.ndarray, noise: np.ndarray = None):
    Uclipped = np.maximum(np.minimum(U, CONTROL_LIMITS), -CONTROL_LIMITS)
    FNED, TXYZ = compute_forces_and_torques_batch_fn(X, Uclipped, params)

    if noise is not None:
        FNED = FNED + noise[0:3].reshape(1, -1)
        TXYZ = TXYZ + noise[3:6].reshape(1, -1)

    X1 = X.copy()
    X1[:, NED_DOT_IDXS] += dt * FNED / params[0]
    X1[:, PQR_IDXS] += dt * TXYZ / np.array([params[1], params[2], params[3]])

    X1[:, NED_IDXS] += dt * X[:, NED_DOT_IDXS]
    X1[:, AXIS_ANGLE_IDXS] += axis_angle_dynamics_update_batch(
        X[:, AXIS_ANGLE_IDXS], X[:, PQR_IDXS] * dt)

    return X1


@njit
def compute_forces_and_torques_batch_fn(X: np.ndarray, U: np.ndarray, params: np.ndarray):
    QUAT = quaternion_from_axis_rotation_batch(X[:, AXIS_ANGLE_IDXS])
    UVW = express_vector_in_quat_frame_batch(
        X[:, NED_DOT_IDXS],
        QUAT,
    )

    UVW0 = UVW[:, 0]
    UVW1 = np.vstack((np.ones(UVW.shape[0]), UVW[:, 1])).T
    UVW2 = np.vstack((np.ones(UVW.shape[0]), UVW[:, 2], U[:, 3])).T

    FXYZ_MINUS_G = np.vstack(
        (
            UVW0 * params[13],
            UVW1 @ params[14:16],
            UVW2 @ params[16:19],
        )
    ).T

    F_NED_MINUS_G = rotate_vector_batch(FXYZ_MINUS_G, QUAT)

    FNED = F_NED_MINUS_G + params[0] * np.array([0, 0, params[19]])

    PQR1 = np.vstack((np.ones(U.shape[0]), X[:, PQR_IDXS[0]], U[:, 0])).T
    PQR2 = np.vstack((np.ones(U.shape[0]), X[:, PQR_IDXS[1]], U[:, 1])).T
    PQR3 = np.vstack((np.ones(U.shape[0]), X[:, PQR_IDXS[2]], U[:, 2])).T

    TXYZ = np.vstack(
        (
            PQR1 @ params[4:7],
            PQR2 @ params[7:10],
            PQR3 @ params[10:13],
        )
    ).T

    return FNED, TXYZ
