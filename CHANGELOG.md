# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-02

### Added
- Initial release of rl-bandits package
- Stationary and non-stationary k-armed bandit environments
- Multiple bandit algorithms:
  - Greedy (sample-average and constant step-size)
  - ε-Greedy (sample-average and constant step-size)
  - UCB (Upper Confidence Bound)
  - Gradient Bandit (preference-based)
- Comprehensive experiment framework:
  - Single and multi-run experiments
  - YAML-based configuration system
  - Enhanced plotting utilities
- Full test suite with 67% coverage
- CI/CD workflow with GitHub Actions
- Professional documentation and examples

### Features
- Modular architecture for easy extension
- Reproducible experiments with proper seeding
- Publication-quality plotting capabilities
- Type hints throughout codebase
- Comprehensive error handling
