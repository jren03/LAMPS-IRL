# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn

import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac

from .core import Agent


# class SACAgent(Agent, nn.Module):
class SACAgent(nn.Module):
    """A Soft-Actor Critic agent.

    This class is a wrapper for
    https://github.com/luisenp/pytorch_sac/blob/master/pytorch_sac/agent/sac.py


    Args:
        (pytorch_sac.SACAgent): the agent to wrap.
    """

    def __init__(self, sac_agent: pytorch_sac.SAC):
        super(SACAgent, self).__init__()
        self.sac_agent = sac_agent

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        with torch.no_grad():
            return self.sac_agent.select_action(
                obs, batched=batched, evaluate=not sample
            )

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass
