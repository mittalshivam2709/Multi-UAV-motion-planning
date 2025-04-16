import torch
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.model(x)

class GAILAgent:
    def __init__(self, policy, discriminator, policy_optimizer, disc_optimizer):
        self.policy = policy
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer

    def update(self, expert_batch, policy_batch):
        exp_obs, exp_act = expert_batch
        pol_obs, pol_act = policy_batch
        exp_labels = torch.ones((len(exp_obs), 1))
        pol_labels = torch.zeros((len(pol_obs), 1))

        disc_loss = nn.BCELoss()(self.discriminator(exp_obs, exp_act), exp_labels) + \
                    nn.BCELoss()(self.discriminator(pol_obs, pol_act), pol_labels)

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        rewards = -torch.log(self.discriminator(pol_obs, pol_act) + 1e-8)
        loss = -(rewards.mean())

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
