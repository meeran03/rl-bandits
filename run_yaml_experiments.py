#!/usr/bin/env python3
"""
Run YAML-configured experiments using the enhanced plotting system.

This script demonstrates how to run the experiment configurations
with the new enhanced plotting capabilities.
"""

import os
import sys
from rlbandits.experiments.yaml_runner import run_yaml_experiment, run_all_experiments


def main():
    """Run YAML experiments with enhanced plotting."""
    print("🎯 RL Bandits YAML Experiment Runner")
    print("=" * 50)
    
    experiments_dir = "rlbandits/experiments"
    
    if len(sys.argv) > 1:
        # Run specific experiment
        experiment_name = sys.argv[1]
        if not experiment_name.endswith('.yaml'):
            experiment_name += '.yaml'
        
        config_path = os.path.join(experiments_dir, experiment_name)
        
        if not os.path.exists(config_path):
            print(f"❌ Experiment file not found: {config_path}")
            print(f"Available experiments:")
            yaml_files = [f for f in os.listdir(experiments_dir) if f.endswith('.yaml')]
            for f in sorted(yaml_files):
                print(f"  - {f}")
            return
        
        print(f"🚀 Running single experiment: {experiment_name}")
        run_yaml_experiment(config_path)
        
    else:
        # Run all experiments
        print("🚀 Running all YAML experiments...")
        run_all_experiments(experiments_dir)
    
    print("\n✅ YAML experiments completed with enhanced plotting!")


if __name__ == "__main__":
    main()
