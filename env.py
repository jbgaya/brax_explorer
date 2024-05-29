import gym

class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def action_space(self):
        return self.env.action_space.n

    def state_space(self):
        return self.env.observation_space.shape[0]