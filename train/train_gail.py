import torch
from agents.gail_agent import GAILAgent, Discriminator
from agents.policy_networks import MLPPolicy
from envs.airsim_env import MultiUAVEnv

def train():
    env = MultiUAVEnv()
    obs_dim, act_dim = 4, 2
    policy = MLPPolicy(obs_dim, act_dim)
    discriminator = Discriminator(obs_dim, act_dim)

    agent = GAILAgent(
        policy, discriminator,
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    )

    for epoch in range(100):
        obs = env.reset()
        acts = [policy(torch.tensor(o).float()).detach() for o in obs]
        next_obs, _, _, _ = env.step(acts)

        expert_obs = torch.stack([torch.tensor(o).float() for o in obs])
        expert_act = torch.stack([torch.tensor(a).float() for a in acts])
        agent.update((expert_obs, expert_act), (expert_obs, expert_act))

if __name__ == '__main__':
    train()
