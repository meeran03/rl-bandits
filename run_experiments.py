#!/usr/bin/env python3
"""
Experiment runner script that recreates the original functionality.

This script runs the same experiments as the original greedy.py file
but using the new modular structure.
"""

from rlbandits import (
    Bandit, NonStationaryBandit,
    GreedyAgent, EpsilonGreedyAgent, UCBAgent, GradientBanditAgent,
    run_multi_experiment, plot_curves, print_summary
)


def main():
    """Run all experiments from the original script."""
    print("🎰 Running RL Bandits Experiments")
    print("=" * 50)
    
    # Experiment 1: Greedy agent on stationary bandit
    print("\n1️⃣ Greedy Agent (Stationary Bandit)")
    greedy = GreedyAgent(k=10)
    curves = run_multi_experiment(
        bandit_class=Bandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0},
        agent=greedy,
        steps=1000,
        runs=200,
        warm_start=True
    )
    print_summary(curves, "Greedy")
    plot_curves(curves, title="Greedy Agent Performance")
    
    # Experiment 2: ε-Greedy agent on stationary bandit
    print("\n2️⃣ ε-Greedy Agent (Stationary Bandit)")
    eg = EpsilonGreedyAgent(k=10, epsilon=0.1)
    curves = run_multi_experiment(
        bandit_class=Bandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0},
        agent=eg,
        steps=1000,
        runs=200,
        warm_start=True
    )
    print_summary(curves, "ε-Greedy")
    plot_curves(curves, title="ε-Greedy Agent Performance")
    
    # Experiment 3: Greedy with constant step-size on non-stationary bandit
    print("\n3️⃣ Greedy (α=0.1) on Non-stationary Bandit")
    greedy2 = GreedyAgent(k=10, alpha=0.1)
    curves = run_multi_experiment(
        bandit_class=NonStationaryBandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0, "walk_std": 0.01},
        agent=greedy2,
        steps=1000,
        runs=200,
        warm_start=False
    )
    print_summary(curves, "Greedy (α=0.1)")
    plot_curves(curves, title="Greedy (α=0.1) on Non-stationary Bandit")
    
    # Experiment 4: ε-Greedy with constant step-size on non-stationary bandit
    print("\n4️⃣ ε-Greedy (α=0.1) on Non-stationary Bandit")
    eg2 = EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=0.1)
    curves = run_multi_experiment(
        bandit_class=NonStationaryBandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0, "walk_std": 0.01},
        agent=eg2,
        steps=1000,
        runs=200,
        warm_start=False
    )
    print_summary(curves, "ε-Greedy (α=0.1)")
    plot_curves(curves, title="ε-Greedy (α=0.1) on Non-stationary Bandit")
    
    # Experiment 5: UCB agent on stationary bandit
    print("\n5️⃣ UCB Agent (Stationary Bandit)")
    ucb = UCBAgent(k=10, c=2.0)
    curves = run_multi_experiment(
        bandit_class=Bandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0},
        agent=ucb,
        steps=1000,
        runs=200,
        warm_start=True
    )
    print_summary(curves, "UCB")
    plot_curves(curves, title="UCB Agent Performance")
    
    # Experiment 6: Gradient Bandit agent on stationary bandit
    print("\n6️⃣ Gradient Bandit Agent (Stationary Bandit)")
    grad = GradientBanditAgent(k=10, alpha=0.1)
    curves = run_multi_experiment(
        bandit_class=Bandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0},
        agent=grad,
        steps=1000,
        runs=200,
        warm_start=False
    )
    print_summary(curves, "Gradient Bandit")
    plot_curves(curves, title="Gradient Bandit Agent Performance")
    
    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()
