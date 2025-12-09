from __future__ import annotations

from updated_research_env import ResearchEnv
from updated_agents import QLearningAgent, SFAgent, BAMDPSRAgent
from updated_experiment import (
    run_experiment,
    plot_metrics,
    sanity_checks,
    evaluate_accuracy,
    plot_comparison,
    collect_traces,
    plot_context_trajectories,
    plot_drift_score,
    plot_state_heatmap,
)


def main():
    sanity_checks()

    env = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=True)
    n_episodes = 200
    max_steps = 20

    results = {}
    accuracies = {}

    results["Q-learning"] = run_experiment(
        env, QLearningAgent, n_episodes, max_steps,
        {"alpha": 0.1, "epsilon": 0.1, "gamma": 0.95},
    )

    results["Successor Features"] = run_experiment(
        env, SFAgent, n_episodes, max_steps,
        {"alpha_sr": 0.3, "alpha_w": 0.3, "gamma": 0.95, "epsilon": 0.1},
    )

    results["Adaptive SR"] = run_experiment(
        env, BAMDPSRAgent, n_episodes, max_steps,
        {
            "alpha_sr": 0.2,
            "alpha_w": 0.2,
            "alpha_t": 0.05,
            "gamma": 0.95,
            "drift_alpha": 1.0,
            "drift_beta": 1.0,
            "drift_threshold": 0.5,
            "explore_steps": 3,
        },
    )

    plot_metrics(results, smoothing=10)

    print("Evaluation results (average return and accuracy on static env):")
    eval_env = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=False)
    for name, res in results.items():
        agent = res["agent"]
        avg_return = agent.evaluate(n_episodes=30, max_steps=max_steps)
        acc = evaluate_accuracy(agent, eval_env, n_episodes=30, max_steps=max_steps)
        accuracies[name] = acc
        print(f"{name}: return={avg_return:.3f}, accuracy={acc:.3f}")

    plot_comparison(results, accuracies)

    print("Collecting traces for Adaptive SR...")
    env_traces = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=True)
    agent, traces = collect_traces(
        env_traces,
        BAMDPSRAgent,
        n_episodes=50,
        max_steps=max_steps,
        agent_kwargs={
            "alpha_sr": 0.2,
            "alpha_w": 0.2,
            "alpha_t": 0.05,
            "gamma": 0.95,
            "drift_alpha": 1.0,
            "drift_beta": 1.0,
            "drift_threshold": 0.5,
            "explore_steps": 3,
        },
    )

    plot_context_trajectories(traces, n_episodes_to_plot=5)
    plot_drift_score(traces, episode_index=0)
    plot_state_heatmap(traces, max_T=max_steps)

    print("Done. Plots saved to current directory.")


if __name__ == "__main__":
    main()