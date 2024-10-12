import torch
import torch.nn as nn
import torch.optim as optim
import gym
from policy_gradient_simple import PolicyNetwork, compute_returns

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

def collect_trajectories(env, policy, gamma=0.99):
    states, actions, rewards = [], [], []
    state, _ = env.reset()  # Updated Gym API

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():  # No gradients needed during sampling
            action_probs = policy(state_tensor)
            action = torch.multinomial(action_probs, 1).item()

        next_state, reward, done, truncated, _ = env.step(action)  # Updated step() return

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)

        if done or truncated:
            break

        state = next_state

    returns = compute_returns(rewards, gamma)
    return states, actions, returns

# Various training methods with different baselines
def train_policy_gradient_constant_baseline(env_name='CartPole-v1', baseline=10, **kwargs):
    _train_with_baseline(env_name, lambda _: baseline, **kwargs)

def train_policy_gradient_optimal_constant_baseline(env_name='CartPole-v1', **kwargs):
    _train_with_baseline(env_name, lambda returns: torch.mean(returns), **kwargs)

def train_policy_gradient_time_dependent_baseline(env_name='CartPole-v1', **kwargs):
    _train_with_baseline(
        env_name,
        lambda returns: torch.cumsum(returns, 0) / torch.arange(1, len(returns) + 1),
        **kwargs
    )

def train_policy_gradient_state_dependent_baseline(env_name='CartPole-v1', lr=1e-2, gamma=0.99, num_episodes=1000):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, action_dim)
    value_net = ValueNetwork(input_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    for episode in range(num_episodes):
        states, actions, returns = collect_trajectories(env, policy, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.cat([value_net(state) for state in states]).squeeze(1)
        advantages = returns - values.detach()

        # Update value network
        value_loss = torch.mean((values - returns) ** 2)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Update policy network
        policy_loss = -torch.sum(
            torch.stack([torch.log(policy(state).squeeze(0)[action]) * G
                         for state, action, G in zip(states, actions, advantages)])
        )
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

    env.close()

def _train_with_baseline(env_name, baseline_fn, lr=1e-2, gamma=0.99, num_episodes=1000):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        states, actions, returns = collect_trajectories(env, policy, gamma)
        baseline = baseline_fn(torch.tensor(returns, dtype=torch.float32))
        advantages = torch.tensor(returns, dtype=torch.float32) - baseline

        # Compute policy loss
        policy_loss = -torch.sum(
            torch.stack([torch.log(policy(state).squeeze(0)[action]) * G
                         for state, action, G in zip(states, actions, advantages)])
        )

        # Update policy network
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {policy_loss.item()}")

    env.close()

if __name__ == "__main__":
    train_policy_gradient_state_dependent_baseline()
