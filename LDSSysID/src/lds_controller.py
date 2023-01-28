import numpy as np
from lds import LDS
from lqr import lqr_ltv
from controller import LinearController


def optimal_controller_lds(model: LDS):
    K, P = lqr_ltv(
        [model.A.copy() for _ in range(model.horizon)]
        if not model.time_varying
        else model.A,
        [model.B.copy() for _ in range(model.horizon)]
        if not model.time_varying
        else model.B,
        model.Q,
        model.R,
        model.Qf,
    )

    return LinearController(
        K,
        P,
        [model.nominal_state for _ in range(model.horizon)],
        [model.nominal_control for _ in range(model.horizon)],
        time_invariant=False,
    )
