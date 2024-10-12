import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def collect_trajectories(env, policy, gamma=0.99):
    states, actions, rewards = [], [], []

    # Reset environment and get the initial state
    state, _ = env.reset()
    done = False

    while not done:
        # Convert state to tensor and get action probabilities from policy
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = policy(state_tensor)

        # Sample an action based on the probabilities
        action = torch.multinomial(action_probs, 1).item()

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Store trajectory data
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # Move to the next state
        state = next_state

    returns = compute_returns(rewards, gamma)
    return states, actions, returns

def compute_returns(rewards, gamma):
    """Compute discounted returns for an episode."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

def get_policy_parameters(policy):
    """Retrieve the policy parameters (weights and biases)."""
    theta = {}
    for name, param in policy.named_parameters():
        theta[name] = param.detach().numpy()  # Convert tensor to NumPy array
        print(f"Parameter {name}: {theta[name].shape}")
    return theta

def train_policy_gradient_simple():
    """Train the policy using the policy gradient method."""
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    for episode in range(1000):
        # Collect trajectories and calculate returns
        states, actions, returns = collect_trajectories(env, policy)

        # Compute policy loss
        log_probs = []
        for state, action in zip(states, actions):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)
            log_prob = torch.log(action_probs.squeeze(0)[action])
            log_probs.append(log_prob)

        # Stack log probabilities and calculate loss
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * returns)  # Policy gradient loss

        # Perform backpropagation and update policy parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}, Loss: {loss.item()}")

    # Retrieve and print the final policy parameters
    print("\nFinal Policy Parameters:")
    theta = get_policy_parameters(policy)

if __name__ == "__main__":
    train_policy_gradient_simple()
