import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class InverseRL:
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim=64):
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.reward_weights = nn.Parameter(torch.randn(feature_dim))
        self.optimizer = optim.Adam(list(self.feature_net.parameters()) + [self.reward_weights])

    def feature_expectations(self, states, actions):
        state_action_pairs = torch.cat([states, actions], dim=1)
        return self.feature_net(state_action_pairs).mean(dim=0)

    def train(self, expert_states, expert_actions, env, num_iterations=100):
        expert_fe = self.feature_expectations(expert_states, expert_actions)

        for iteration in range(num_iterations):
            random_states = torch.FloatTensor(np.random.rand(1000, expert_states.shape[1]))
            random_actions = torch.FloatTensor(np.random.rand(1000, expert_actions.shape[1]))
            random_fe = self.feature_expectations(random_states, random_actions)

            expert_reward = torch.dot(self.reward_weights, expert_fe)
            random_reward = torch.dot(self.reward_weights, random_fe)
            loss = torch.max(torch.tensor(0.0), random_reward - expert_reward + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}")

    def get_reward(self, state, action):
        state_action_pair = torch.cat([state, action], dim=0)
        feature = self.feature_net(state_action_pair)
        return torch.dot(self.reward_weights, feature).item()
