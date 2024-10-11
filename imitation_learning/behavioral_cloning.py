import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BehavioralCloning:
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

    def train(self, expert_states, expert_actions, epochs=100, batch_size=32):
        expert_states = np.array(expert_states)  # Convert to NumPy array
        expert_actions = np.array(expert_actions)  # Convert to NumPy array

        # Convert to PyTorch tensors
        expert_states_tensor = torch.tensor(expert_states, dtype=torch.float32)
        expert_actions_tensor = torch.tensor(expert_actions, dtype=torch.long)  # Assuming actions are indices

        dataset = torch.utils.data.TensorDataset(expert_states, expert_actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for states, actions in dataloader:
                self.optimizer.zero_grad()
                pred_actions = self.policy(states)
                loss = self.loss_fn(pred_actions, actions)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    def predict(self, state):
        with torch.no_grad():
            return self.policy(state)
