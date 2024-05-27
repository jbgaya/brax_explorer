import torch
from torch.distributions import Categorical

class REINFORCE:
    def __init__(self, policy, optimizer, gamma=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state,_info = env.reset()
            log_probs = []
            rewards = []
            done = False
            while not done:
                state = torch.tensor(state, dtype=torch.float32)
                probs = self.policy(state)
                m = Categorical(probs)
                action = m.sample()
                state, reward, truncated, terminated, _ = env.step(action.item())
                done = truncated | terminated 
                log_probs.append(m.log_prob(action))
                rewards.append(reward)

            # Calculate returns and update policy
            returns, G = [], 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            policy_loss = sum([-log_prob * G for log_prob, G in zip(log_probs, returns)])
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()