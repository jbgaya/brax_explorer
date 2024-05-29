import torch.optim as optim
from utils import instantiate_class, create_environment
import argparse
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description='Run the reinforcement learning model.')
    parser.add_argument('-c', '--config', type=str, default='configs/mlp.yaml',
                        help='Path to the configuration file.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Instantiate environment
    env = create_environment(config.env.name)

    # Prepare model parameters dynamically based on environment
    model_config = config.model
    model_config.parameters.state_size = env.state_space()
    model_config.parameters.action_size = env.action_space()

    # Instantiate model, optimizer, logger, and algorithm
    model = instantiate_class('model', model_config.name, **model_config.parameters)
    optimizer_class = getattr(optim, config.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **config.optimizer.parameters)
    logger = instantiate_class('logger', config.logger.name)
    algo = instantiate_class('algorithm', config.algorithm.name, policy=model, optimizer=optimizer, logger=logger, **config.algorithm.parameters)

    # Training
    algo.train(env)
    env.close()

if __name__ == "__main__":
    main()
