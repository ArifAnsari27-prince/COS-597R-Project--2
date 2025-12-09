from __future__ import annotations

from updated_bamdp_sr import NonStationaryMDP, BayesAdaptiveSRAgent


def make_simple_env():
    """
    Create a tiny NonStationaryMDP with 2 contexts on a 5-state chain.
    Context 0: reward at right, Context 1: reward at left.
    """
    n_states = 5
    n_actions = 2

    # Context 0: normal chain, reward at right end
    trans0 = []
    rew0 = []
    for s in range(n_states):
        ns0 = max(0, s - 1)
        ns1 = min(n_states - 1, s + 1)
        trans0.append([ns0, ns1])
        rew0.append([
            1.0 if ns0 == n_states - 1 else 0.0,
            1.0 if ns1 == n_states - 1 else 0.0,
        ])

    # Context 1: reversed semantics, reward at left end
    trans1 = []
    rew1 = []
    for s in range(n_states):
        ns0 = min(n_states - 1, s + 1)
        ns1 = max(0, s - 1)
        trans1.append([ns0, ns1])
        rew1.append([
            1.0 if ns0 == 0 else 0.0,
            1.0 if ns1 == 0 else 0.0,
        ])

    contexts = [
        {"transitions": trans0, "rewards": rew0},
        {"transitions": trans1, "rewards": rew1},
    ]

    env = NonStationaryMDP(n_states=n_states, n_actions=n_actions, contexts=contexts, initial_state=2)
    return env


def main():
    env = make_simple_env()
    agent = BayesAdaptiveSRAgent(env, n_states=env.n_states, n_actions=env.n_actions)

    print("Training GRU-based BAMDP SR agent for a few episodes...")
    for ep in range(10):
        ret = agent.run_episode(max_steps=30)
        print(f"Episode {ep}: return={ret:.3f}, steps={len(agent.last_trace)}")

    avg_return = agent.evaluate(n_episodes=20, max_steps=30)
    print(f"Average evaluation return: {avg_return:.3f}")


if __name__ == "__main__":
    main()