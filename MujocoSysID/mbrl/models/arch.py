from gym.spaces import Discrete
from torch import nn
import numpy as np
import torch
from mbrl.util.nn_utils import create_mlp, init_ortho
from termcolor import cprint


class Discriminator(nn.Module):
    def __init__(self, env, tanh_disc=False, clip=False):
        super(Discriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        if tanh_disc:
            net = create_mlp(
                self.obs_dim + self.action_dim,
                1,
                self.net_arch,
                activation_fn=nn.ReLU,
                output_activation_fn=nn.Tanh,
            )
        else:
            net = create_mlp(
                self.obs_dim + self.action_dim,
                1,
                self.net_arch,
                activation_fn=nn.ReLU,
                output_activation_fn=None,
            )
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

        if clip:
            cprint("Clipping discriminator", color="magenta", attrs=["bold"])
            self.clip = True
            self.clip_min, self.clip_max = -40, 40
        else:
            self.clip = False

    def forward(self, inputs):
        output = self.net(inputs)
        if self.clip:
            output = torch.clamp(output, -40, 40)
        return output.view(-1)


class DiscriminatorEnsemble(nn.Module):
    def __init__(self, env, n_discriminators=7, tanh_disc=False):
        super(DiscriminatorEnsemble, self).__init__()
        self.ensemble = nn.ModuleList(
            [Discriminator(env) for _ in range(n_discriminators, tanh_disc=tanh_disc)]
        )

    def forward(self, inputs):
        outputs = torch.stack([disc(inputs) for disc in self.ensemble])
        return torch.min(outputs, dim=0)[0]
