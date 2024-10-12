from policy_gradient_simple import train_policy_gradient_simple
from policy_gradient_baseline import (
    train_policy_gradient_constant_baseline,
    train_policy_gradient_optimal_constant_baseline,
    train_policy_gradient_time_dependent_baseline,
    train_policy_gradient_state_dependent_baseline,
)

if __name__ == "__main__":
    print("Running Simple Policy Gradient...")
    train_policy_gradient_simple()

    print("\nRunning Policy Gradient with Constant Baseline...")
    train_policy_gradient_constant_baseline()

    print("\nRunning Policy Gradient with Optimal Constant Baseline...")
    train_policy_gradient_optimal_constant_baseline()

    print("\nRunning Policy Gradient with Time-Dependent Baseline...")
    train_policy_gradient_time_dependent_baseline()

    print("\nRunning Policy Gradient with State-Dependent Baseline...")
    train_policy_gradient_state_dependent_baseline()
