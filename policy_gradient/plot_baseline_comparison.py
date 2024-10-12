import matplotlib.pyplot as plt
import numpy as np
from policy_gradient_baseline import (
    train_policy_gradient_constant_baseline,
    train_policy_gradient_optimal_constant_baseline,
    train_policy_gradient_time_dependent_baseline,
    train_policy_gradient_state_dependent_baseline,
)

def run_experiments():
    num_episodes = 1000

    # Store the results for plotting
    constant_baseline_rewards = []
    optimal_constant_baseline_rewards = []
    time_dependent_baseline_rewards = []
    state_dependent_baseline_rewards = []

    print("Running Constant Baseline...")
    constant_baseline_rewards = train_policy_gradient_constant_baseline(num_episodes=num_episodes)

    print("Running Optimal Constant Baseline...")
    optimal_constant_baseline_rewards = train_policy_gradient_optimal_constant_baseline(num_episodes=num_episodes)

    print("Running Time-Dependent Baseline...")
    time_dependent_baseline_rewards = train_policy_gradient_time_dependent_baseline(num_episodes=num_episodes)

    print("Running State-Dependent Baseline...")
    state_dependent_baseline_rewards = train_policy_gradient_state_dependent_baseline(num_episodes=num_episodes)

    return (constant_baseline_rewards, optimal_constant_baseline_rewards,
            time_dependent_baseline_rewards, state_dependent_baseline_rewards)

def plot_results(results):
    labels = [
        "Constant Baseline",
        "Optimal Constant Baseline",
        "Time-Dependent Baseline",
        "State-Dependent Baseline"
    ]

    plt.figure(figsize=(12, 6))

    for idx, (label, rewards) in enumerate(zip(labels, results)):
        if rewards:  # Check if rewards is not empty
            plt.plot(rewards, label=label)
        else:
            print(f"Warning: No rewards for {label}")

    plt.title("Policy Gradient Baselines Comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)
