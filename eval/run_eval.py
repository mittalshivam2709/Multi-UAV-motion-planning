from envs.airsim_env import MultiUAVEnv
from agents.policy_networks import MLPPolicy
import torch

def run_eval():
    env = MultiUAVEnv()
    policy = MLPPolicy(obs_dim=4, act_dim=2)
    policy.load_state_dict(torch.load("saved_policy.pt"))

    obs = env.reset()
    done = False
    while not done:
        actions = [policy(torch.tensor(o).float()).detach().numpy() for o in obs]
        obs, _, done, _ = env.step(actions)

if __name__ == "__main__":
    run_eval()
