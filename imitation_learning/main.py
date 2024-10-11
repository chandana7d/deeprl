import torch
import gym
from behavioral_cloning import BehavioralCloning
from dagger import DAgger
from inverse_rl import InverseRL
import config
import numpy as np

def collect_expert_data(env, expert_policy):
    expert_states = []
    expert_actions = []

    state = env.reset()

    done = False
    while not done:
        action = expert_policy(state)  # Assuming expert_policy returns a valid action
        result = env.step(action)  # Capture all returned values
        print(result)  # Check what is being returned
        state, reward, done, info = result[:4]  # Modify this based on the output

        expert_states.append(state)
        expert_actions.append(action)

    return expert_states, expert_actions


def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy(torch.FloatTensor(state)).argmax().item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    return total_rewards / num_episodes

def main():
    torch.manual_seed(config.SEED)
    env = gym.make(config.ENV_NAME)

    # Collect expert data
    expert_states, expert_actions = collect_expert_data(env, config.expert_policy)

    # Behavioral Cloning
    print("Training Behavioral Cloning...")
    bc_model = BehavioralCloning(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM)
    bc_model.train(expert_states, expert_actions, epochs=config.BC_EPOCHS)
    bc_reward = evaluate_policy(env, bc_model.predict)
    print(f"Behavioral Cloning Average Reward: {bc_reward:.2f}")

    # DAgger
    print("\nTraining DAgger...")
    dagger_model = DAgger(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM)
    dagger_model.train(env, config.expert_policy, num_iterations=config.DAGGER_ITERATIONS,
                       num_episodes=config.DAGGER_EPISODES, epochs=config.DAGGER_EPOCHS)
    dagger_reward = evaluate_policy(env, dagger_model.predict)
    print(f"DAgger Average Reward: {dagger_reward:.2f}")

    # Inverse RL
    print("\nTraining Inverse RL...")
    irl_model = InverseRL(config.STATE_DIM, config.ACTION_DIM, config.FEATURE_DIM, config.HIDDEN_DIM)
    irl_model.train(expert_states, expert_actions, env, num_iterations=config.IRL_ITERATIONS)
    print("IRL training completed. Reward function learned.")

if __name__ == "__main__":
    main()
