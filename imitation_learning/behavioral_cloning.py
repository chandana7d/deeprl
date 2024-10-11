import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BehavioralCloningModel:
    def __init__(self, input_size, num_classes, hidden_dim):
        # Define your neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),  # First layer
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # Output layer
        )
        self.loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, expert_states, expert_actions, epochs):
        # Create a DataLoader for batching
        dataset = TensorDataset(torch.tensor(expert_states, dtype=torch.float32),
                                torch.tensor(expert_actions, dtype=torch.long))  # Convert actions to long
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()  # Zero gradients

                # Forward pass
                pred_actions = self.model(batch_states)  # Shape: (batch_size, num_classes)

                # Compute loss
                loss = self.loss_fn(pred_actions, batch_actions)  # batch_actions should be class indices

                # Backward pass and optimization step
                loss.backward()
                self.optimizer.step()

                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
