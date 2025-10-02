#!/usr/bin/env python3
"""
Basic usage example for rl-bandits library.

This script demonstrates how to use the library to run bandit experiments
and compare different algorithms.
"""

import sys
import os

# Add parent directory to path to import rlbandits
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlbandits import (
    Bandit, NonStationaryBandit,
    GreedyAgent, EpsilonGreedyAgent, UCBAgent, GradientBanditAgent,
    run_multi_experiment, print_summary
)


def main():
    """Run basic bandit experiments."""
    print("🎰 RL Bandits Library - Basic Usage Example")
    print("=" * 50)
    
    # Experiment 1: Compare agents on stationary bandit
    print("\n📊 Experiment 1: Stationary Bandit Comparison")
    print("-" * 40)
    
    agents = {
        "Greedy": GreedyAgent(k=10),
        "ε-Greedy (0.01)": EpsilonGreedyAgent(k=10, epsilon=0.01),
        "ε-Greedy (0.1)": EpsilonGreedyAgent(k=10, epsilon=0.1),
        "UCB (c=2)": UCBAgent(k=10, c=2.0),
        "Gradient Bandit": GradientBanditAgent(k=10, alpha=0.1),
    }
    
    for name, agent in agents.items():
        results = run_multi_experiment(
            bandit_class=Bandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0},
            agent=agent,
            steps=1000,
            runs=100,
            warm_start=True
        )
        print_summary(results, name)
    
    # Experiment 2: Non-stationary bandit with different update rules
    print("\n📈 Experiment 2: Non-stationary Bandit")
    print("-" * 40)
    
    ns_agents = {
        "Greedy (sample-avg)": GreedyAgent(k=10, alpha=None),
        "Greedy (α=0.1)": GreedyAgent(k=10, alpha=0.1),
        "ε-Greedy (sample-avg)": EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=None),
        "ε-Greedy (α=0.1)": EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=0.1),
    }
    
    for name, agent in ns_agents.items():
        results = run_multi_experiment(
            bandit_class=NonStationaryBandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0, "walk_std": 0.01},
            agent=agent,
            steps=1000,
            runs=100,
            warm_start=False
        )
        print_summary(results, name)
    
    print("\n✅ Experiments completed!")
    print("\nTip: Use plot_curves(results) to visualize the performance curves.")


if __name__ == "__main__":
    main()
