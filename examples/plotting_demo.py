#!/usr/bin/env python3
"""
Enhanced plotting demonstration for rl-bandits library.

This script shows the improved plotting capabilities including:
- Single agent plots
- Multi-agent comparisons  
- Parameter studies
- Saving plots to files
"""

import sys
import os

# Add parent directory to path to import rlbandits
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlbandits import (
    Bandit, NonStationaryBandit,
    GreedyAgent, EpsilonGreedyAgent, UCBAgent, GradientBanditAgent,
    run_multi_experiment, plot_curves, plot_comparison, plot_parameter_study
)


def demo_single_agent_plot():
    """Demonstrate single agent plotting."""
    print("📊 Demo 1: Single Agent Plot")
    print("-" * 30)
    
    agent = EpsilonGreedyAgent(k=10, epsilon=0.1)
    results = run_multi_experiment(
        bandit_class=Bandit,
        bandit_kwargs={"k": 10, "reward_std": 1.0},
        agent=agent,
        steps=1000,
        runs=50,
        warm_start=True
    )
    
    # Basic plot
    plot_curves(results, title="ε-Greedy Agent Performance")
    
    # Enhanced plot with save
    plot_curves(
        results, 
        title="ε-Greedy Agent (Enhanced)",
        save_path="epsilon_greedy_performance.png",
        figsize=(14, 7),
        show=False  # Don't show, just save
    )


def demo_multi_agent_comparison():
    """Demonstrate multi-agent comparison plotting."""
    print("\n🔄 Demo 2: Multi-Agent Comparison")
    print("-" * 35)
    
    agents = {
        "Greedy": GreedyAgent(k=10),
        "ε-Greedy (0.01)": EpsilonGreedyAgent(k=10, epsilon=0.01),
        "ε-Greedy (0.1)": EpsilonGreedyAgent(k=10, epsilon=0.1),
        "UCB (c=2)": UCBAgent(k=10, c=2.0),
    }
    
    results = {}
    for name, agent in agents.items():
        print(f"Running {name}...")
        results[name] = run_multi_experiment(
            bandit_class=Bandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0},
            agent=agent,
            steps=1000,
            runs=50,
            warm_start=True
        )
    
    # Plot comparison
    plot_comparison(
        results, 
        title="Bandit Algorithm Comparison",
        save_path="algorithm_comparison.png",
        figsize=(16, 8)
    )


def demo_parameter_study():
    """Demonstrate parameter study plotting."""
    print("\n🔬 Demo 3: Parameter Study (Epsilon Values)")
    print("-" * 45)
    
    epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = {}
    
    for eps in epsilon_values:
        print(f"Running ε={eps}...")
        agent = EpsilonGreedyAgent(k=10, epsilon=eps)
        results[f"ε={eps}"] = run_multi_experiment(
            bandit_class=Bandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0},
            agent=agent,
            steps=1000,
            runs=30,  # Fewer runs for speed
            warm_start=True
        )
    
    # Plot parameter study
    plot_parameter_study(
        results,
        parameter_name="Epsilon",
        title="ε-Greedy Parameter Study",
        save_path="epsilon_parameter_study.png"
    )


def demo_ucb_parameter_study():
    """Demonstrate UCB confidence parameter study."""
    print("\n🎯 Demo 4: UCB Confidence Parameter Study")
    print("-" * 45)
    
    c_values = [0.5, 1.0, 1.414, 2.0, 3.0, 5.0]  # √2 ≈ 1.414
    results = {}
    
    for c in c_values:
        print(f"Running c={c}...")
        agent = UCBAgent(k=10, c=c)
        label = f"c={c}" if c != 1.414 else "c=√2"
        results[label] = run_multi_experiment(
            bandit_class=Bandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0},
            agent=agent,
            steps=1000,
            runs=30,
            warm_start=True
        )
    
    plot_parameter_study(
        results,
        parameter_name="UCB Confidence",
        title="UCB Confidence Parameter Study",
        save_path="ucb_parameter_study.png"
    )


def demo_nonstationary_comparison():
    """Demonstrate comparison on non-stationary environment."""
    print("\n📈 Demo 5: Non-stationary Environment Comparison")
    print("-" * 50)
    
    agents = {
        "Greedy (sample-avg)": GreedyAgent(k=10, alpha=None),
        "Greedy (α=0.1)": GreedyAgent(k=10, alpha=0.1),
        "ε-Greedy (sample-avg)": EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=None),
        "ε-Greedy (α=0.1)": EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=0.1),
    }
    
    results = {}
    for name, agent in agents.items():
        print(f"Running {name}...")
        results[name] = run_multi_experiment(
            bandit_class=NonStationaryBandit,
            bandit_kwargs={"k": 10, "reward_std": 1.0, "walk_std": 0.01},
            agent=agent,
            steps=1000,
            runs=50,
            warm_start=False
        )
    
    plot_comparison(
        results,
        title="Non-stationary Bandit: Sample-Average vs Constant Step-Size",
        save_path="nonstationary_comparison.png",
        figsize=(16, 8)
    )


def main():
    """Run all plotting demonstrations."""
    print("🎨 RL Bandits Enhanced Plotting Demo")
    print("=" * 50)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        demo_single_agent_plot()
        demo_multi_agent_comparison()
        demo_parameter_study()
        demo_ucb_parameter_study()
        demo_nonstationary_comparison()
        
        print("\n✅ All plotting demos completed!")
        print("📁 Check the 'results/' directory for saved plots.")
        
    except ImportError as e:
        if "matplotlib" in str(e):
            print("❌ matplotlib is required for plotting demos.")
            print("Install with: pip install matplotlib")
        else:
            raise


if __name__ == "__main__":
    main()
