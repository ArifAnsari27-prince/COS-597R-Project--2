"""
Script to run experiments comparing different agents on the research environment.

This script trains a Q‑learning agent, a successor‑feature agent, and an
adaptive BAMDP SR agent on the same non‑stationary environment.  It
records metrics such as episode returns and training time, evaluates
each trained agent, and plots the results.  Use this script as a
starting point for more extensive experimentation.
"""

from __future__ import annotations

from research_env import ResearchEnv
from agents import QLearningAgent, SFAgent, BAMDPSRAgent
from experiment import run_experiment, plot_metrics, sanity_checks, evaluate_accuracy
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Sanity checks on environment and agents
    sanity_checks()
    # Create a research environment with drift occurring at a random time in each episode
    env = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=True)
    n_episodes = 200
    max_steps = 20
    results = {}
    accuracies = {}
    # Q‑learning baseline
    results['Q‑learning'] = run_experiment(env, QLearningAgent, n_episodes, max_steps,
                                          {'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.95})
    # Successor‑feature baseline
    results['Successor Features'] = run_experiment(env, SFAgent, n_episodes, max_steps,
                                                  {'alpha_sr': 0.3, 'alpha_w': 0.3, 'gamma': 0.95, 'epsilon': 0.1})
    # Adaptive BAMDP SR agent
    results['Adaptive SR'] = run_experiment(env, BAMDPSRAgent, n_episodes, max_steps,
                                           {'alpha_sr': 0.2, 'alpha_w': 0.2, 'alpha_t': 0.05,
                                            'gamma': 0.95, 'drift_alpha': 1.0, 'drift_beta': 1.0,
                                            'drift_threshold': 0.5, 'explore_steps': 3})
    # Evaluate policy accuracy and average return for each agent on a static environment
    print("Evaluating trained agents on a static context (no drift) for returns and accuracy...")
    eval_env = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=False)
    for name, res in results.items():
        agent = res['agent']
        avg_return = agent.evaluate(n_episodes=30, max_steps=max_steps)
        acc = evaluate_accuracy(eval_env, agent, n_episodes=30, max_steps=max_steps)
        accuracies[name] = acc
        print(f"{name}: return={avg_return:.3f}, accuracy={acc:.3f}")
    # Plot combined metrics
    plot_metrics(results, accuracies, smoothing=10)


if __name__ == '__main__':
    main()