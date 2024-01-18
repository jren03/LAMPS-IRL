from gym.spaces import Discrete
from torch import nn
import numpy as np
import torch
from mbrl.util.common import create_mlp, init_ortho


class Discriminator(nn.Module):
    def __init__(self, env):
        super(Discriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(self.obs_dim + self.action_dim, 1, self.net_arch, nn.ReLU)
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

    def forward(self, inputs):
        output = self.net(inputs)
        return output.view(-1)


class DiscriminatorEnsemble(nn.Module):
    def __init__(self, env, n_discriminators=7, reduction="min", ema_decay=0.99):
        super(DiscriminatorEnsemble, self).__init__()
        self.ensemble = nn.ModuleList(
            [Discriminator(env) for _ in range(n_discriminators)]
        )
        self.reduction = reduction
        print(f"Using {reduction} reduction for discriminator ensemble")

    def forward(self, inputs):
        outputs = torch.stack([disc(inputs) for disc in self.ensemble])
        if self.reduction == "min":
            return torch.min(outputs, dim=0)[0]
        elif self.reduction == "mean":
            return torch.mean(outputs, dim=0)
        elif self.reduction == "median":
            return torch.median(outputs, dim=0)[0]
        elif self.reduction == "max":
            return torch.max(outputs, dim=0)[0]
        elif self.reduction == "log":
            red = torch.clamp(
                -torch.log(1 - torch.mean(outputs, dim=0) + 1e-8), min=-4, max=15
            )
            if torch.isnan(red).any():
                breakpoint()
                raise ValueError("NaN in discriminator output")
            if torch.isinf(red).any():
                breakpoint()
                raise ValueError("Inf in discriminator output")
            return red
        else:
            raise NotImplementedError
