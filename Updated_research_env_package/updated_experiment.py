from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Type, Dict, Any, Optional, Tuple

from updated_agents import QLearningAgent, SFAgent, BAMDPSRAgent, BaseAgent
from updated_research_env import ResearchEnv
import time


def run_experiment(
    env: ResearchEnv,
    agent_cls: Type[BaseAgent],
    n_episodes: int,
    max_steps: int,
    agent_kwargs: Dict[str, float],
) -> Dict[str, Any]:
    """
    Train an agent on the environment and record metrics.
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
        "returns": episode_returns,
        "durations": episode_durations,
        "training_time": total_time,
        "agent": agent,
    }


def collect_traces(
    env: ResearchEnv,
    agent_cls: Type[BaseAgent],
    n_episodes: int,
    max_steps: int,
    agent_kwargs: Dict[str, float],
) -> Tuple[BaseAgent, List[List[Dict[str, Any]]]]:
    """
    Train an agent and collect per-episode trajectories (last_trace).
    """
    agent = agent_cls(env, **agent_kwargs)
    traces: List[List[Dict[str, Any]]] = []
    for _ in range(n_episodes):
        _ = agent.train_episode(max_steps=max_steps)
        if getattr(agent, "last_trace", None):
            traces.append(list(agent.last_trace))
    return agent, traces


def evaluate_accuracy(agent: BaseAgent, env: ResearchEnv, n_episodes: int = 30, max_steps: int = 100) -> float:
    """
    Estimate a simple 'policy accuracy' metric: fraction of actions that
    move in the direction of the true rewarding goal for the current context.
    """
    total_correct = 0
    total_actions = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        t = 0
        ctx = env.current_context
        while not done and t < max_steps:
            action = agent.act(state, exploit=True)
            if ctx == 0:
                correct = (action == 1)  # right
            else:
                # in ctx=1, action 1 moves left (good) under our definition
                correct = (action == 1)
            total_correct += int(correct)
            total_actions += 1

            state, _, done, info = env.step(action)
            ctx = info.get("context", ctx)
            t += 1

    return total_correct / max(1, total_actions)


def plot_metrics(results: Dict[str, Dict[str, Any]], smoothing: int = 10) -> None:
    """
    Plot smoothed episode returns and training times for multiple agents.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        returns = np.array(res["returns"])
        if smoothing > 1 and len(returns) >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.convolve(returns, kernel, mode="valid")
            xs = np.arange(len(smoothed))
        else:
            smoothed = returns
            xs = np.arange(len(returns))
        ax.plot(xs, smoothed, label=name)
    ax.set_title("Smoothed episode returns")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig("returns_plot.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    times = [results[name]["training_time"] for name in names]
    ax2.bar(names, times)
    ax2.set_title("Total training time")
    ax2.set_ylabel("Seconds")
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=45)
    fig2.tight_layout()
    fig2.savefig("training_times_plot.png")
    plt.close(fig2)


def plot_context_trajectories(traces: List[List[Dict[str, Any]]], n_episodes_to_plot: int = 5) -> None:
    """
    Plot state trajectories for episodes where context changed at least once.
    """
    episodes_with_drift = []
    for ep_trace in traces:
        ctxs = [step["context"] for step in ep_trace]
        if len(set(ctxs)) > 1:
            episodes_with_drift.append(ep_trace)

    if not episodes_with_drift:
        print("No episodes with context change found in traces.")
        return

    episodes_with_drift = episodes_with_drift[:n_episodes_to_plot]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, ep_trace in enumerate(episodes_with_drift):
        ts = np.array([step["t"] for step in ep_trace])
        states = np.array([step["state"] for step in ep_trace])
        contexts = np.array([step["context"] for step in ep_trace])
        explores = np.array([step["explore"] for step in ep_trace], dtype=bool)

        initial_ctx = contexts[0]
        drift_indices = np.where(contexts != initial_ctx)[0]
        drift_t = ts[drift_indices[0]] if len(drift_indices) > 0 else None

        ax.plot(ts, states, label=f"Episode {idx}", alpha=0.8)
        ax.scatter(ts[explores], states[explores],
                   marker="o", s=40, edgecolor="k", facecolor="none")

        if drift_t is not None:
            ax.axvline(drift_t, color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Time step")
    ax.set_ylabel("State index")
    ax.set_title("State trajectories in episodes with context change\n"
                 "(red = context switch, circles = exploration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("context_trajectories.png")
    plt.close(fig)


def plot_drift_score(traces: List[List[Dict[str, Any]]], episode_index: int = 0) -> None:
    """
    Plot zeta_t, SR error and model error for a single episode.
    """
    if episode_index >= len(traces):
        print("episode_index out of range.")
        return

    ep_trace = traces[episode_index]
    ts = np.array([step["t"] for step in ep_trace])
    zetas = np.array([step["zeta"] for step in ep_trace], dtype=float)
    sr_err = np.array([step["sr_error"] for step in ep_trace], dtype=float)
    model_err = np.array([step["model_error"] for step in ep_trace], dtype=float)
    contexts = np.array([step["context"] for step in ep_trace])
    explores = np.array([step["explore"] for step in ep_trace], dtype=bool)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(ts, zetas, label="zeta_t")
    ax[0].scatter(ts[explores], zetas[explores],
                  color="red", label="explore", zorder=3)
    ax[0].set_ylabel("zeta_t")
    ax[0].legend()

    ax[1].plot(ts, sr_err, label="SR error ||delta_SR||^2")
    ax[1].plot(ts, model_err, label="model error", linestyle="--")
    ax[1].set_ylabel("Error")
    ax[1].legend()

    ax[2].step(ts, contexts, where="post")
    ax[2].set_ylabel("Context")
    ax[2].set_xlabel("Time step")

    fig.suptitle(f"Drift score and errors – episode {episode_index}")
    fig.tight_layout()
    fig.savefig("drift_score_episode.png")
    plt.close(fig)


def plot_state_heatmap(traces: List[List[Dict[str, Any]]], max_T: Optional[int] = None) -> None:
    """
    Heatmap of visitation counts over (time, state) across all episodes.
    """
    max_state = max(max(step["state"] for step in ep) for ep in traces)
    max_time = max(max(step["t"] for step in ep) for ep in traces)
    if max_T is not None:
        max_time = min(max_time, max_T)

    counts = np.zeros((max_time + 1, max_state + 1))
    for ep in traces:
        for step in ep:
            t = step["t"]
            s = step["state"]
            if t <= max_time:
                counts[t, s] += 1

    plt.figure(figsize=(8, 5))
    plt.imshow(counts.T, aspect="auto", origin="lower")
    plt.colorbar(label="visits")
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.title("State visitation heatmap")
    plt.tight_layout()
    plt.savefig("state_heatmap.png")
    plt.close()


def sanity_checks() -> None:
    """
    Simple smoke tests for environment and agents.
    """
    env = ResearchEnv(n_states=4, n_actions=2, drift_in_episode=False)
    s0 = env.reset()
    assert 0 <= s0 < env.n_states
    ns, r, done, info = env.step(0)
    assert 0 <= ns < env.n_states
    assert isinstance(r, float)

    q_agent = QLearningAgent(env, alpha=0.5, epsilon=0.2, gamma=0.9)
    for _ in range(5):
        stats = q_agent.train_episode(max_steps=5)
        assert isinstance(stats.returns, float)

    sf_agent = SFAgent(env, alpha_sr=0.3, alpha_w=0.3, gamma=0.9, epsilon=0.1)
    for _ in range(5):
        stats = sf_agent.train_episode(max_steps=5)
        assert isinstance(stats.returns, float)

    print("Sanity checks passed.")


def interactive_env_demo(env: ResearchEnv, n_steps: int = 10, render_mode: str = "human") -> None:
    """
    Manually step through the environment for intuition.
    """
    state = env.reset()
    env.render(mode=render_mode)
    for _ in range(n_steps):
        try:
            action = int(input("Enter action (0=left, 1=right): "))
        except Exception:
            print("Invalid input, defaulting to 0.")
            action = 0
        if not (0 <= action < env.n_actions):
            print("Invalid action, defaulting to 0.")
            action = 0
        state, reward, done, info = env.step(action)
        env.render(mode=render_mode)
        print(f"Reward: {reward}\n")
        if done:
            print("Episode ended.")
            break


def plot_comparison(results: Dict[str, Dict[str, Any]], accuracies: Dict[str, float]) -> None:
    """
    Combined comparison plot: returns, training time, accuracy.
    """
    names = list(results.keys())
    avg_returns = [np.mean(results[name]["returns"]) for name in names]
    times = [results[name]["training_time"] for name in names]
    accs = [accuracies.get(name, 0.0) for name in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x - width, avg_returns, width, label="avg return")
    ax1.set_ylabel("Average return")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(x, accs, width, label="accuracy", color="tab:orange")
    ax2.set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig("comparison_metrics.png")
    plt.close(fig)