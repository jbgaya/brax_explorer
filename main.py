import torch.optim as optim
from env import Environment
from model import PolicyNetwork
from algorithm import REINFORCE

def main():
    env = Environment('CartPole-v1')
    policy = PolicyNetwork(env.state_space(), env.action_space())
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    algo = REINFORCE(policy, optimizer)

    algo.train(env, num_episodes=1000)
    env.close()

if __name__ == "__main__":
    main()