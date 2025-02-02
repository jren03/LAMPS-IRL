import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from mbrl.third_party.pytorch_sac_pranz24.model import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
)
from mbrl.third_party.pytorch_sac_pranz24.utils import (
    hard_update,
    soft_update,
    linear_schedule,
)

from mbrl.util.oadam import OAdam
from mbrl.util.common import PrintColors as PC


# class SAC(object):
class SAC(nn.Module):
    def __init__(self, num_inputs, relabel_samples, action_space, args):
        super(SAC, self).__init__()
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None or args.target_entropy == -1:
                    print(PC.BOLD + "Using automatic entropy tuning" + PC.ENDC)
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            else:
                print(PC.WARNING + "WARNING: Entropy is not being tuned." + PC.ENDC)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.f_net = None
        self.updates_made = 0
        self.relabel_samples = relabel_samples
        self.total_timesteps = args.total_timesteps
        print(PC.BOLD + f"SAC initialized with {self.total_timesteps=}" + PC.ENDC)

    def add_f_net(self, f_net):
        self.f_net = f_net
        if self.relabel_samples:
            print(
                PC.WARNING
                + "WARNING: SAC is relabeling samples with f_net. This is not standard SAC."
                + PC.ENDC
            )
        else:
            print(
                PC.WARNING
                + "WARNING: SAC is NOT relabeling samples with f_net. This is standard SAC."
                + PC.ENDC
            )

    def reset_optimizers(self, optim_oadam=False):
        if optim_oadam:
            self.critic_optim = OAdam(self.critic.parameters(), lr=self.args.lr)
            self.policy_optim = OAdam(self.policy.parameters(), lr=self.args.lr)
        else:
            self.critic_optim = Adam(self.critic.parameters(), lr=self.args.lr)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.args.lr)
        # self.alpha_optim = Adam([self.log_alpha], lr=self.args.lr)
        self.get_schedule_fn = linear_schedule(self.args.lr)
        self.updates_made = 0

    def step_lr(self):
        optimizers = [self.critic_optim, self.policy_optim]
        for optim in optimizers:
            for param_group in optim.param_groups:
                progress_remaining = max(
                    1 - self.updates_made / self.total_timesteps, 1e-8
                )
                param_group["lr"] = self.get_schedule_fn(progress_remaining)

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def estimate_value(self, state):
        _, _, action = self.policy.sample(state)
        q1, q2 = self.critic(state, action)

        return torch.min(q1, q2)

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        reward_batch = self._relabel_with_f_net(
            state_batch,
            action_batch,
            torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1),
        )
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.updates_made % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # if logger is not None:
        #     logger.log("train/batch_reward", reward_batch.mean(), updates)
        #     logger.log("train_critic/loss", qf_loss, updates)
        #     logger.log("train_actor/loss", policy_loss, updates)
        #     if self.automatic_entropy_tuning:
        #         logger.log("train_actor/target_entropy", self.target_entropy, updates)
        #     else:
        #         logger.log("train_actor/target_entropy", 0, updates)
        #     logger.log("train_actor/entropy", -log_pi.mean(), updates)
        #     logger.log("train_alpha/loss", alpha_loss, updates)
        #     logger.log("train_alpha/value", self.alpha, updates)

        # return (
        #     qf1_loss.item(),
        #     qf2_loss.item(),
        #     policy_loss.item(),
        #     alpha_loss.item(),
        #     alpha_tlogs.item(),
        # )

    def adv_update_parameters(
        self,
        memory,
        expert_memory,
        batch_size,
        updates,
        logger=None,
        reverse_mask=False,
    ):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        (
            expert_state_batch,
            expert_action_batch,
            *_,
        ) = expert_memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        reward_batch = self._relabel_with_f_net(
            state_batch,
            action_batch,
            torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1),
        )

        expert_state_batch = torch.FloatTensor(expert_state_batch).to(self.device)
        expert_action_batch = torch.FloatTensor(expert_action_batch).to(self.device)

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            try:
                next_state_action, next_state_log_pi, _ = self.policy.sample(
                    next_state_batch
                )
            except:
                breakpoint()
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(expert_state_batch)

        qf1_pi, qf2_pi = self.critic(expert_state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        with torch.no_grad():
            qf1_expert, qf2_expert = self.critic_target(
                expert_state_batch, expert_action_batch
            )
            min_qf_expert = torch.min(qf1_expert, qf2_expert)

        policy_loss = (
            (self.alpha * log_pi) + (min_qf_expert - min_qf_pi)
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.updates_made % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # if logger is not None:
        #     logger.log("train/batch_reward", reward_batch.mean(), updates)
        #     logger.log("train_critic/loss", qf_loss, updates)
        #     logger.log("train_actor/loss", policy_loss, updates)
        #     if self.automatic_entropy_tuning:
        #         logger.log("train_actor/target_entropy", self.target_entropy, updates)
        #     else:
        #         logger.log("train_actor/target_entropy", 0, updates)
        #     logger.log("train_actor/entropy", -log_pi.mean(), updates)
        #     logger.log("train_alpha/loss", alpha_loss, updates)
        #     logger.log("train_alpha/value", self.alpha, updates)

        # return (
        #     qf1_loss.item(),
        #     qf2_loss.item(),
        #     policy_loss.item(),
        #     alpha_loss.item(),
        #     alpha_tlogs.item(),
        # )

    # relabel rewards with f_net
    @torch.no_grad()
    def _relabel_with_f_net(self, state_batch, action_batch, reward_batch):
        # relabel rewards with f_net
        if self.f_net is not None and self.relabel_samples:
            sa_pair = torch.cat((state_batch, action_batch), dim=1)
            reward_batch = -self.f_net(sa_pair).reshape(reward_batch.shape)
        return reward_batch

    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cuda:0")
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
