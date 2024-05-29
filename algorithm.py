import torch
from torch.distributions import Categorical
from copy import deepcopy

class REINFORCE:
    def __init__(self, policy, optimizer, logger=None, gamma=0.99, num_episodes = 1_000):
        self.policy = policy
        self.optimizer = optimizer
        self.logger = logger  # Default to basic Logger if none provided
        self.gamma = gamma
        self.num_episodes = num_episodes

    def train(self, env):
        total_rewards = []
        for episode in range(self.num_episodes):
            state, info = env.reset()
            log_probs = []
            rewards = []
            done = False
            t = 0
            while not done:
                state = torch.tensor(state, dtype=torch.float32)
                probs = self.policy(state, t)
                m = Categorical(probs)
                action = m.sample()
                state, reward, truncated, terminated, _ = env.step(action.item())
                done = truncated | terminated 
                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                t += 1
            
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            self.update_policy(log_probs, rewards)

            if self.logger:
                self.logger.log(episode_reward=total_reward)

        self.logger.close()

    def update_policy(self, log_probs, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        policy_loss = sum([-log_prob * G for log_prob, G in zip(log_probs, returns)])
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()




class REINFORCEWithOldPolicy:
    def __init__(self, policy, optimizer, logger=None, gamma=0.99, num_episodes=1000, refresh_policy_every_n_steps=10):
        self.policy = policy
        self.optimizer = optimizer
        self.logger = logger
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.refresh_policy_every_n_steps = refresh_policy_every_n_steps
        self.old_policy = deepcopy(policy)
        self.steps_since_refresh = 0

    def train(self, env):
        total_rewards = []
        step_count = 0
        for episode in range(self.num_episodes):
            state, info = env.reset()
            log_probs = []
            rewards = []
            ratios = []
            done = False
            t = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                probs = self.policy(state_tensor, t)

                m = Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)

                with torch.no_grad():
                    old_probs = self.old_policy(state_tensor, t)
                    old_m = Categorical(old_probs)
                    old_log_prob = old_m.log_prob(action)
                
                
                state, reward, truncated, terminated, _ = env.step(action.item())
                done = truncated | terminated
                log_probs.append(log_prob)
                rewards.append(reward)
                # Importance sampling ratio
                ratio = torch.exp(- log_prob + old_log_prob)
                ratios.append(ratio)
                t += 1

                step_count += 1
                if step_count % self.refresh_policy_every_n_steps == 0:
                    self.old_policy.load_state_dict(self.policy.state_dict())

            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            self.update_policy(log_probs, rewards, ratios)

            if self.logger:
                self.logger.log(episode_reward=total_reward)

        if self.logger:
            self.logger.close()

    def update_policy(self, log_probs, rewards, ratios):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        policy_loss = sum([-ratio * log_prob * G for ratio, log_prob, G in zip(ratios, log_probs, returns)])
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
