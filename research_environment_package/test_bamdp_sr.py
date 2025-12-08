"""
Simple demonstration of the Bayes‑Adaptive SR agent on a toy non‑stationary
environment.  Two contexts are defined with different transition and reward
structures.  The agent is trained over a number of episodes and then
evaluated to illustrate its ability to adapt to latent context changes.  To
run this script simply execute it with Python.  Training progress and final
performance are printed to the console.
"""

from __future__ import annotations

# Try to import the torch‑based implementation.  If PyTorch is not installed
# this will fail and we will fall back to the NumPy implementation.
try:
    import torch  # noqa: F401
    from bamdp_sr import NonStationaryMDP, BAMDPAgent  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    from bamdp_sr_numpy import NonStationaryMDP, BAMDPNumpyAgent


def make_toy_env() -> NonStationaryMDP:
    """Create a simple environment with two contexts.

    States: 0,1,2,3,4; Actions: 0=left, 1=right.
    Context 0: transitions encourage moving right; reward of +1 on the
    terminal state 4.  Context 1: transitions encourage moving left; reward
    of +1 on the terminal state 0.  The agent must infer whether it should
    go left or right based on the observed rewards and transitions.
    """
    n_states = 5
    n_actions = 2
    # Context 0
    transitions0 = {
        0: {0: 0, 1: 1},
        1: {0: 0, 1: 2},
        2: {0: 1, 1: 3},
        3: {0: 2, 1: 4},
        4: {0: 3, 1: 4},  # terminal on the right
    }
    rewards0 = {
        0: {0: 0.0, 1: 0.0},
        1: {0: 0.0, 1: 0.0},
        2: {0: 0.0, 1: 0.0},
        3: {0: 0.0, 1: 0.0},
        4: {0: 1.0, 1: 1.0},
    }
    # Context 1: transitions flipped and reward at the left
    transitions1 = {
        0: {0: 1, 1: 0},
        1: {0: 2, 1: 0},
        2: {0: 3, 1: 1},
        3: {0: 4, 1: 2},
        4: {0: 4, 1: 3},  # terminal on the left
    }
    rewards1 = {
        0: {0: 1.0, 1: 1.0},
        1: {0: 0.0, 1: 0.0},
        2: {0: 0.0, 1: 0.0},
        3: {0: 0.0, 1: 0.0},
        4: {0: 0.0, 1: 0.0},
    }
    contexts = [
        {'transitions': transitions0, 'rewards': rewards0},
        {'transitions': transitions1, 'rewards': rewards1},
    ]
    return NonStationaryMDP(n_states=n_states, n_actions=n_actions, contexts=contexts, initial_state=2, gamma=0.9)


def main():
    env = make_toy_env()
    if TORCH_AVAILABLE:
        agent = BAMDPAgent(env, latent_dim=8, enc_hidden=64, sr_hidden=128, policy_hidden=128,
                           gamma=0.9, lr=1e-3, sr_loss_weight=1.0, value_loss_weight=0.5, entropy_weight=0.01)
    else:
        agent = BAMDPNumpyAgent(env, z_dim=2, gamma=0.9, lr=0.05, sr_loss_weight=1.0, value_loss_weight=0.5,
                                entropy_weight=0.01)
    n_episodes = 300
    print_every = 50
    returns = []
    for episode in range(1, n_episodes + 1):
        R = agent.train_episode(max_steps=30)
        returns.append(R)
        if episode % print_every == 0:
            avg_R = sum(returns[-print_every:]) / print_every
            print(f"Episode {episode}/{n_episodes}, average return over last {print_every}: {avg_R:.3f}")
    # Evaluate without learning
    avg_eval_return = agent.evaluate(n_episodes=20, max_steps=30)
    print(f"\nEvaluation average return (no learning): {avg_eval_return:.3f}")


if __name__ == "__main__":
    main()