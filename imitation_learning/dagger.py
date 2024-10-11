import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DAgger:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.policy.parameters())
        self.loss_fn = nn.MSELoss()

    def train(self, env, expert_policy, num_iterations=10, num_episodes=100, epochs=100):
        dataset = []

        for iteration in range(num_iterations):
            for _ in range(num_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.predict(torch.FloatTensor(state))
                    next_state, _, done, _ = env.step(action.numpy())
                    expert_action = expert_policy(state)
                    dataset.append((state, expert_action))
                    state = next_state

            states, actions = zip(*dataset)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)

            for epoch in range(epochs):
                self.optimizer.zero_grad()
                pred_actions = self.policy(states)
                loss = self.loss_fn(pred_actions, actions)
                loss.backward()
                self.optimizer.step()

            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}")

    def predict(self, state):
        with torch.no_grad():
            return self.policy(state)
