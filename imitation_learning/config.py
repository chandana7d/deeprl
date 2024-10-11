import gym
import torch
import numpy as np

# Environment configuration
ENV_NAME = 'CartPole-v1'

# General settings
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
env = gym.make(ENV_NAME)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
HIDDEN_DIM = 64
FEATURE_DIM = 16

# Training parameters
BC_EPOCHS = 100
DAGGER_ITERATIONS = 10
DAGGER_EPISODES = 100
DAGGER_EPOCHS = 100
IRL_ITERATIONS = 100

# Expert policy (example)
def expert_policy(state):
    # Check if the state has enough elements before accessing state[2]
    if len(state) > 2:
        return 0 if state[2] < 0 else 1
    else:
        return 0  # Default action if state has fewer elements
