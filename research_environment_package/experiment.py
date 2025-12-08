"""
Experimental harness for adaptive SR research.

This module contains functions to run experiments comparing different
reinforcement learning agents on the research environment.  It also
provides helper routines to plot results and perform simple sanity
checks on the environment and agents.  The goal is to facilitate
backtesting, benchmarking and visualisation of new algorithms.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Type, Dict
from agents import QLearningAgent, SFAgent, BAMDPSRAgent, BaseAgent, EpisodeStats
from research_env import ResearchEnv
import time


def run_experiment(env: ResearchEnv, agent_cls: Type[BaseAgent], n_episodes: int, max_steps: int,
                   agent_kwargs: Dict[str, float]) -> Dict[str, List[float]]:
    """Train an agent on the environment and record metrics.

    Args:
        env: The research environment (will be reset each episode).
        agent_cls: The agent class to instantiate.
        n_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        agent_kwargs: Additional keyword arguments passed to the agent.
    Returns:
        A dictionary containing the episode returns, durations and total
        training time.
    """
    agent = agent_cls(env, **agent_kwargs)
    episode_returns: List[float] = []
    episode_durations: List[float] = []
    start_time = time.time()
    for _ in range(n_episodes):
        stats = agent.train_episode(max_steps)
        episode_returns.append(stats.returns)
        episode_durations.append(stats.duration)
    total_time = time.time() - start_time
    return {
        'returns': episode_returns,
        'durations': episode_durations,
        'training_time': total_time,
        'agent': agent,
    }


def evaluate_accuracy(env: ResearchEnv, agent: BaseAgent, n_episodes: int, max_steps: int) -> float:
    """Compute the proportion of correct actions taken by an agent.

    A correct action is defined as choosing the direction that leads
    towards the rewarding end of the chain for the current hidden
    context.  For context 0 (reward on the right), action 1 is deemed
    correct; for context 1 (reward on the left), action 0 is correct.

    Parameters
    ----------
    env : ResearchEnv
        The environment on which to evaluate.
    agent : BaseAgent
        The trained agent to evaluate (must implement ``act``).
    n_episodes : int
        Number of evaluation episodes.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    float
        The average proportion of correct actions across all steps and
        episodes.
    """
    total_correct = 0
    total_steps = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        t = 0
        while not done and t < max_steps:
            # Determine optimal action: 0 for left context (reward on left), 1 for right context
            current_context = env.current_context
            optimal_action = 1 if current_context == 0 else 0
            # agent chooses action greedily
            action = agent.act(state, exploit=True)
            if action == optimal_action:
                total_correct += 1
            total_steps += 1
            state, _, done, _ = env.step(action)
            t += 1
    return total_correct / max(total_steps, 1)


def plot_metrics(results: Dict[str, Dict[str, List[float]]], accuracies: Dict[str, float], smoothing: int = 10) -> None:
    """Plot returns, training times and accuracy for multiple agents.

    Parameters
    ----------
    results : dict
        Mapping from agent names to dictionaries returned by
        ``run_experiment``.
    accuracies : dict
        Mapping from agent names to scalar accuracy scores (0–1) as
        returned by ``evaluate_accuracy``.
    smoothing : int, optional
        Window size for moving average smoothing of episode returns.
    """
    # Plot smoothed returns
    fig, (ax_ret, ax_time, ax_acc) = plt.subplots(1, 3, figsize=(15, 4))
    for name, res in results.items():
        returns = np.array(res['returns'])
        if smoothing > 1 and len(returns) >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.convolve(returns, kernel, mode='valid')
            x = range(len(smoothed))
        else:
            smoothed = returns
            x = range(len(returns))
        ax_ret.plot(x, smoothed, label=name)
    ax_ret.set_title('Smoothed episode returns')
    ax_ret.set_xlabel('Episode')
    ax_ret.set_ylabel('Return')
    ax_ret.legend()
    # Bar chart for training times
    names = list(results.keys())
    times = [results[name]['training_time'] for name in names]
    ax_time.bar(names, times)
    ax_time.set_title('Total training time')
    ax_time.set_ylabel('Seconds')
    ax_time.set_xticks(range(len(names)))
    ax_time.set_xticklabels(names, rotation=45)
    # Bar chart for accuracies
    acc_vals = [accuracies.get(name, 0.0) for name in names]
    ax_acc.bar(names, acc_vals)
    ax_acc.set_title('Policy accuracy')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xticks(range(len(names)))
    ax_acc.set_xticklabels(names, rotation=45)
    fig.tight_layout()
    # Save figure to file
    fig.savefig('comparison_metrics.png')
    plt.close(fig)


def sanity_checks() -> None:
    """Run simple tests to validate environment and agents."""
    env = ResearchEnv(n_states=4, n_actions=2, drift_in_episode=False)
    # Check environment transitions
    s0 = env.reset()
    assert 0 <= s0 < env.n_states, 'Reset returned invalid state'
    # Step once
    ns, r, done, info = env.step(0)
    assert 0 <= ns < env.n_states, 'Next state invalid'
    assert isinstance(r, float), 'Reward not float'
    # Run a few steps with Q‑learning and SF agents to ensure no errors
    q_agent = QLearningAgent(env, alpha=0.5, epsilon=0.2, gamma=0.9)
    for _ in range(5):
        stats = q_agent.train_episode(max_steps=5)
        assert isinstance(stats.returns, float), 'QLearning returns not float'
    sf_agent = SFAgent(env, alpha_sr=0.3, alpha_w=0.3, gamma=0.9, epsilon=0.1)
    for _ in range(5):
        stats = sf_agent.train_episode(max_steps=5)
        assert isinstance(stats.returns, float), 'SFAgent returns not float'
    print('Sanity checks passed.')


def interactive_env_demo(env: ResearchEnv, n_steps: int = 10) -> None:
    """Interactively display the environment and prompt user for actions.

    This helper allows a user to step through the environment manually
    to get an intuition of its behaviour.  Actions are entered via the
    console: 0 for left and 1 for right.  The current state and context
    are rendered at each step.
    """
    state = env.reset()
    # Use text rendering initially
    env.render()
    for _ in range(n_steps):
        try:
            action = int(input('Enter action (0=left, 1=right): '))
        except Exception:
            print('Invalid input, defaulting to 0.')
            action = 0
        if not (0 <= action < env.n_actions):
            print('Invalid action, defaulting to 0.')
            action = 0
        state, reward, done, info = env.step(action)
        # Render both text and, if available, a plot
        env.render()  # text
        try:
            env.render(mode='plot')  # plot representation if matplotlib is available
        except Exception:
            pass
        print(f'Reward: {reward}\n')
        if done:
            print('Episode ended.')
            break
