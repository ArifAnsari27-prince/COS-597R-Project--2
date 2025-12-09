from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Tuple, Optional, Dict, Any, List

try:
    from gym import spaces
except Exception:
    spaces = None


class ResearchEnv:
    """
    Chain environment with hidden contexts and optional drift.

    There are n_states states indexed 0..n_states-1 arranged in a line.
    Two hidden contexts determine which side gives reward and how actions
    map to movements.

    Context 0 (right-reward):
        - Rewards +1 when the agent reaches the rightmost state.
    Context 1 (left-reward):
        - Rewards +1 when the agent reaches the leftmost state.

    The agent does not observe the context.  Context can either:
        - be sampled once per episode (stationary within episode), or
        - drift once within an episode at a random or scheduled time.
    """

    def __init__(
        self,
        n_states: int = 6,
        n_actions: int = 2,
        drift_in_episode: bool = False,
        drift_prob: float = 0.0,
        drift_schedule: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        assert n_actions >= 2, "At least two actions required."
        self.n_states = n_states
        self.n_actions = n_actions
        self.drift_in_episode = drift_in_episode
        self.drift_prob = drift_prob
        self.drift_schedule = drift_schedule
        self.rng = np.random.default_rng(seed)

        self.contexts = self._create_contexts()
        self.current_context: int = 0
        self.drift_time: Optional[int] = None
        self.state: int = 0
        self.t: int = 0
        self.episode_length: int = n_states * 2  # arbitrary upper bound

        if spaces is not None:
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Discrete(self.n_states)

    def _create_contexts(self) -> Dict[int, Dict[str, Dict[Tuple[int, int], Tuple[int, float]]]]:
        """
        Create transition and reward dictionaries for both contexts.
        """
        contexts: Dict[int, Dict[str, Dict[Tuple[int, int], Tuple[int, float]]]] = {}

        # Context 0: actions interpreted as (0=left, 1=right) with reward at rightmost.
        trans0: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            left = max(0, s - 1)
            right = min(self.n_states - 1, s + 1)
            trans0[(s, 0)] = (left, 0.0)
            trans0[(s, 1)] = (right, 0.0)

        reward0: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns, _ = trans0[(s, a)]
                reward0[(s, a)] = (ns, 1.0 if ns == self.n_states - 1 else 0.0)

        contexts[0] = {"trans": trans0, "reward": reward0}

        # Context 1: swap left/right semantics; reward at leftmost.
        trans1: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            left = max(0, s - 1)
            right = min(self.n_states - 1, s + 1)
            # reversed mapping: action 0 moves right, action 1 moves left
            trans1[(s, 0)] = (right, 0.0)
            trans1[(s, 1)] = (left, 0.0)

        reward1: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns, _ = trans1[(s, a)]
                reward1[(s, a)] = (ns, 1.0 if ns == 0 else 0.0)

        contexts[1] = {"trans": trans1, "reward": reward1}
        return contexts

    def reset(self) -> int:
        """
        Reset environment and sample new context and drift time.
        """
        self.current_context = self.rng.integers(0, 2)
        self.state = self.rng.integers(0, self.n_states)
        self.t = 0

        if self.drift_schedule is not None and len(self.drift_schedule) > 0:
            self.drift_time = int(self.rng.choice(self.drift_schedule))
        elif self.drift_in_episode:
            self.drift_time = int(self.rng.integers(1, self.episode_length))
        else:
            self.drift_time = None

        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        """
        assert 0 <= action < self.n_actions

        # Possibly drift context
        if self.drift_in_episode and self.drift_time is not None and self.t == self.drift_time:
            self.current_context = 1 - self.current_context
        elif not self.drift_in_episode and self.drift_prob > 0.0:
            if self.rng.random() < self.drift_prob:
                self.current_context = 1 - self.current_context

        trans = self.contexts[self.current_context]["trans"]
        reward_table = self.contexts[self.current_context]["reward"]

        next_state, _ = trans[(self.state, action)]
        _, reward = reward_table[(self.state, action)]

        self.state = next_state
        self.t += 1
        done = self.t >= self.episode_length

        info = {"context": self.current_context, "t": self.t}
        return self.state, float(reward), done, info

    def render(self, mode: str = "human", ax: Any = None) -> Optional[str]:
        """
        Render the current state.
        """
        chain = ["o"] * self.n_states
        goal_left = "L" if self.current_context == 1 else "l"
        goal_right = "R" if self.current_context == 0 else "r"
        chain[0] = goal_left
        chain[-1] = goal_right
        chain[self.state] = "A"

        out = f"t={self.t} context={self.current_context} | {''.join(chain)}"

        if mode == "ansi":
            return out
        elif mode == "human":
            print(out)
            return None
        elif mode == "plot":
            import matplotlib.pyplot as plt

            xs = np.arange(self.n_states)
            heights = np.ones_like(xs)
            colors = ["lightgray"] * self.n_states
            colors[0] = "tab:blue" if self.current_context == 1 else "lightblue"
            colors[-1] = "tab:orange" if self.current_context == 0 else "moccasin"
            colors[self.state] = "tab:red"

            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 1.5))
                created_fig = True
            else:
                created_fig = False

            ax.bar(xs, heights, color=colors)
            ax.set_yticks([])
            ax.set_xticks(xs)
            ax.set_xlabel(f"t={self.t}, ctx={self.current_context}")
            if created_fig:
                plt.tight_layout()
                plt.show()
            return None
        else:
            raise ValueError(f"Unknown render mode: {mode}")