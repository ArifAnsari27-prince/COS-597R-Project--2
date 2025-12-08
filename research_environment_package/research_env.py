"""
Research environment for evaluating adaptive successor‑representation (SR) agents.

This module defines a simple non‑stationary Markov decision process (MDP)
with two hidden contexts and optional drift within an episode.  The
environment supports discrete states and actions, returns a reward and
indicator of termination.  It exposes a standard API with `reset()` and
`step()` methods similar to OpenAI Gym, making it easy to integrate with
different reinforcement learning agents.  A visual `render()` method is
provided for interactive inspection of the environment state.

Contexts:
    • Context 0 (Right‑reward): transitions move the agent towards the right
      end of a chain; a reward of +1 is received only upon reaching the
      rightmost state.
    • Context 1 (Left‑reward): transitions move the agent towards the left
      end of a chain; a reward of +1 is received only upon reaching the
      leftmost state.

By default a context is chosen uniformly at random at the start of each
episode.  Optionally, the context can drift within an episode: a drift
time is sampled uniformly from the episode length and, once reached, the
context switches to the other setting.  This forces agents to detect
non‑stationarity and adapt their policies accordingly.

Usage example:

```python
>>> env = ResearchEnv(n_states=6, n_actions=2, drift_in_episode=True)
>>> state = env.reset()
>>> done = False
>>> while not done:
...     action = np.random.randint(env.n_actions)
...     next_state, reward, done, info = env.step(action)
...     env.render()
```

This environment is intentionally simple yet supports research on context
switching and adaptive RL algorithms.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict


class ResearchEnv:
    """A chain environment with hidden contexts and optional drift.

    This class implements a simple Markov decision process with two hidden
    contexts.  Each context defines a deterministic transition function on
    a one‑dimensional chain and rewards only when the agent reaches one of
    the terminal states.  The environment is compatible with the OpenAI
    Gym API: it exposes ``action_space`` and ``observation_space``
    attributes so that agents expecting a ``gym.Env`` can interact
    seamlessly.

    Parameters
    ----------
    n_states : int, optional
        Number of discrete states in the chain (including the endpoints).
    n_actions : int, optional
        Number of discrete actions.  Two actions are required: action 0
        moves the agent towards the left end, and action 1 moves it
        towards the right end.  Additional actions are currently ignored.
    drift_in_episode : bool, optional
        If ``True``, the hidden context may switch once at a random
        timestep within each episode.  If ``False``, the context may still
        drift but according to a per‑step probability given by
        ``drift_prob``.
    drift_prob : float, optional
        When ``drift_in_episode`` is ``False``, this sets the
        probability of a context switch at each time step.  A value of
        zero disables random drift.
    drift_schedule : list[int] or None, optional
        A list of time indices at which the context will deterministically
        switch during an episode.  When provided, it overrides
        ``drift_in_episode`` and ``drift_prob``.  Use this to create
        multiple drifts within an episode for robustness experiments.
    seed : int or None, optional
        Seed for the environment's random number generator.

    Notes
    -----
    The environment chooses an initial context uniformly from {0, 1} at
    the start of each episode.  When drift occurs (either at a scheduled
    time or at random), the context flips between 0 and 1.  Terminal
    states are not absorbing: after receiving a reward, the agent can
    continue moving; episodes terminate after ``episode_length`` steps.
    """

    def __init__(self, n_states: int = 6, n_actions: int = 2,
                 drift_in_episode: bool = False, drift_prob: float = 0.0,
                 drift_schedule: Optional[list[int]] = None,
                 seed: Optional[int] = None) -> None:
        # Validate inputs
        assert n_actions >= 2, "At least two actions required."
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.drift_in_episode = bool(drift_in_episode)
        self.drift_prob = float(drift_prob)
        self.drift_schedule = None
        if drift_schedule is not None:
            # Ensure the schedule is sorted and convert to ints
            self.drift_schedule = sorted(int(x) for x in drift_schedule)
        # Random number generator
        self.rng = np.random.default_rng(seed)
        # Build the transition and reward tables for each context
        self.contexts = self._create_contexts()
        # Current hidden context (0 or 1)
        self.current_context: int = 0
        # Next predetermined drift index if using drift_schedule
        self._drift_idx: int = 0
        # Single drift time if drift_in_episode and no schedule
        self.drift_time: Optional[int] = None
        # Current state and timestep
        self.state: int = 0
        self.t: int = 0
        # Episode length: each state may be visited at most twice
        self.episode_length: int = self.n_states * 2
        # Define Gym spaces for compatibility
        try:
            import gym
            from gym import spaces
            self.action_space = spaces.Discrete(self.n_actions)
            # Observations are discrete state indices
            self.observation_space = spaces.Discrete(self.n_states)
        except Exception:
            # If gym is not installed, define placeholders
            self.action_space = None
            self.observation_space = None
    def _create_contexts(self) -> Dict[int, Dict[str, Dict[Tuple[int, int], Tuple[int, float]]]]:
        """Create transition and reward dictionaries for both contexts."""
        contexts: Dict[int, Dict[str, Dict[Tuple[int, int], Tuple[int, float]]]] = {}
        # Context 0: move right, reward at rightmost state
        trans0: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            # action 0: move left (unless at leftmost)
            next_state_left = max(0, s - 1)
            # action 1: move right (unless at rightmost)
            next_state_right = min(self.n_states - 1, s + 1)
            # transitions do not depend on context for simplicity
            trans0[(s, 0)] = (next_state_left, 0.0)
            trans0[(s, 1)] = (next_state_right, 0.0)
        # reward: +1 at rightmost state
        reward0: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state, _ = trans0[(s, a)]
                reward0[(s, a)] = (next_state, 1.0 if next_state == self.n_states - 1 else 0.0)
        contexts[0] = {'trans': trans0, 'reward': reward0}
        # Context 1: move left, reward at leftmost state
        trans1: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            next_state_left = max(0, s - 1)
            next_state_right = min(self.n_states - 1, s + 1)
            # reversed mapping: action 0 moves right, action 1 moves left
            trans1[(s, 0)] = (next_state_right, 0.0)
            trans1[(s, 1)] = (next_state_left, 0.0)
        reward1: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state, _ = trans1[(s, a)]
                reward1[(s, a)] = (next_state, 1.0 if next_state == 0 else 0.0)
        contexts[1] = {'trans': trans1, 'reward': reward1}
        return contexts
    def reset(self) -> int:
        """
        Reset the environment to start a new episode.

        A new hidden context is sampled uniformly from {0,1}, the agent is
        placed at a random state on the chain, and the timestep counter is
        reset.  If a deterministic drift schedule is provided, the drift
        index is reset; otherwise a single random drift time may be
        sampled depending on ``drift_in_episode``.

        Returns
        -------
        int
            The initial state index.
        """
        # Sample initial context and state
        self.current_context = int(self.rng.integers(0, 2))
        self.state = int(self.rng.integers(0, self.n_states))
        self.t = 0
        # Reset drift schedule index or random drift time
        if self.drift_schedule is not None:
            self._drift_idx = 0
            self.drift_time = None
        elif self.drift_in_episode:
            # choose a random drift time strictly within the episode
            self.drift_time = int(self.rng.integers(1, self.episode_length))
        else:
            self.drift_time = None
        return self.state
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: integer in {0..n_actions-1}.
        Returns:
            next_state, reward, done, info.
        """
        assert 0 <= action < self.n_actions
        # Determine drift: scheduled, single random, or per‑step random
        if self.drift_schedule is not None:
            # If current timestep matches the next scheduled drift, flip context
            if self._drift_idx < len(self.drift_schedule) and self.t == self.drift_schedule[self._drift_idx]:
                self.current_context = 1 - self.current_context
                self._drift_idx += 1
        elif self.drift_in_episode and self.drift_time is not None and self.t == self.drift_time:
            # single scheduled drift within the episode
            self.current_context = 1 - self.current_context
        elif not self.drift_in_episode and self.drift_prob > 0.0:
            # random drift based on probability
            if float(self.rng.random()) < self.drift_prob:
                self.current_context = 1 - self.current_context
        # transitions
        trans = self.contexts[self.current_context]['trans']
        reward_table = self.contexts[self.current_context]['reward']
        next_state, _ = trans[(self.state, action)]
        _, reward = reward_table[(self.state, action)]
        self.state = next_state
        self.t += 1
        # done if reached last time step or absorbing state
        done = self.t >= self.episode_length
        return self.state, reward, done, {'context': self.current_context, 't': self.t}
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Render mode.  If ``'human'`` (default), prints a textual
            representation to stdout.  If ``'ansi'``, returns the string
            representation.  If ``'plot'``, attempts to draw a simple
            visualization using ``matplotlib``.  Invalid modes fall back
            to ``'human'``.

        Returns
        -------
        Optional[str]
            When ``mode='ansi'``, the textual representation is returned.
            Otherwise, returns ``None``.
        """
        chain = ['o'] * self.n_states
        # mark agent
        chain[self.state] = 'A'
        # mark goals based on current context; uppercase indicates the goal with reward
        goal_left = 'L' if self.current_context == 1 else 'l'
        goal_right = 'R' if self.current_context == 0 else 'r'
        chain[0] = goal_left
        chain[-1] = goal_right
        line = f"t={self.t} context={self.current_context} | {''.join(chain)}"
        if mode == 'ansi':
            return line
        elif mode == 'plot':
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 1))
                # colours: left reward red, right reward blue, agent green, other grey
                colours = []
                for idx, c in enumerate(chain):
                    if idx == self.state:
                        colours.append('green')
                    elif idx == 0:
                        colours.append('red' if self.current_context == 1 else 'lightgrey')
                    elif idx == self.n_states - 1:
                        colours.append('blue' if self.current_context == 0 else 'lightgrey')
                    else:
                        colours.append('lightgrey')
                ax.bar(range(self.n_states), [1] * self.n_states, color=colours, edgecolor='black')
                ax.set_xticks(range(self.n_states))
                ax.set_xticklabels(range(self.n_states))
                ax.set_yticks([])
                ax.set_xlabel('State index')
                ax.set_title(f"t={self.t} context={self.current_context}")
                fig.tight_layout()
                plt.show()
                return None
            except Exception:
                # fallback to human if matplotlib not available
                print(line)
                return None
        else:
            # human mode: print line
            print(line)
            return None
