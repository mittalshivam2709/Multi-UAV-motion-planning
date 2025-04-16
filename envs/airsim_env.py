import numpy as np

class MultiUAVEnv:
    def __init__(self):
        self.n_uavs = 3
        self.state_dim = 4
        self.action_dim = 2

    def reset(self):
        self.states = [np.random.randn(self.state_dim) for _ in range(self.n_uavs)]
        return self.states

    def step(self, actions):
        for i in range(self.n_uavs):
            vx, vy = actions[i]
            self.states[i][0] += vx * 0.1
            self.states[i][1] += vy * 0.1
            self.states[i][2] = vx
            self.states[i][3] = vy
        return self.states, 0.0, False, {}
