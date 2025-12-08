"""
Agents for research environment.

This module defines a set of reinforcement learning agents used to
benchmark adaptive successor‑representation algorithms.  Three types of
agents are provided:

1. **QLearningAgent**: a tabular Q‑learning agent that treats the
   environment as stationary.  Updates its Q‑table using the standard
   temporal difference rule.  This serves as a simple baseline.

2. **SFAgent**: a successor‑feature agent that maintains a linear SR
   representation M(s,a) and a reward weight vector w.  It updates
   successor features via temporal difference learning and learns the
   reward weights by linear regression on the observed rewards.  This
   agent does not incorporate latent context and thus performs poorly
   when the environment drifts.

3. **BAMDPSRAgent**: an adaptive SR agent inspired by the alternative
   formulation in the project report【338746102192806†L144-L165】.  It maintains an SR
   network, a transition model T(s,a) to estimate uncertainty, and a
   simple controller that monitors prediction errors and decides when
   to switch between exploitation and exploration.  When the drift
   score exceeds a threshold, the agent temporarily explores by
   choosing random actions; otherwise it exploits the learned policy.
   The agent updates its SR representation, reward weights, and
   transition model online.

All agents expose a common interface with `train_episode` and
`evaluate` methods for interaction with the research environment.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import time

from research_env import ResearchEnv


@dataclass
class EpisodeStats:
    """Container for episode statistics."""
    returns: float
    duration: float


class BaseAgent:
    """Base class for reinforcement learning agents."""
    def __init__(self, env: ResearchEnv, gamma: float = 0.95):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = gamma
    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        raise NotImplementedError
    def evaluate(self, n_episodes: int = 10, max_steps: int = 100) -> float:
        total = 0.0
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            t = 0
            discount = 1.0
            while not done and t < max_steps:
                # greedy action
                action = self.act(state, exploit=True)
                next_state, reward, done, _ = self.env.step(action)
                total += discount * reward
                discount *= self.gamma
                state = next_state
                t += 1
        return total / n_episodes
    def act(self, state: int, exploit: bool = False) -> int:
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    """Tabular Q‑learning agent."""
    def __init__(self, env: ResearchEnv, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.95):
        super().__init__(env, gamma)
        self.alpha = alpha
        self.epsilon = epsilon
        # Q‑table initialised to zeros
        self.Q = np.zeros((self.n_states, self.n_actions))
    def act(self, state: int, exploit: bool = False) -> int:
        if not exploit and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))
    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0
        while not done and t < max_steps:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            # update Q
            best_next = np.max(self.Q[next_state])
            td_error = reward + self.gamma * best_next - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1
        return EpisodeStats(total_return, time.time() - start_time)


class SFAgent(BaseAgent):
    """Successor‑feature agent with linear reward model.

    Maintains successor features M(s,a) ∈ ℝ^{n_states} and weight vector
    w ∈ ℝ^{n_states}.  Q(s,a) = M(s,a) · w.  Learns M via TD learning
    (successor feature Bellman equation) and w via linear regression on
    observed rewards.  Does not adapt to context changes, so may
    perform poorly when the environment drifts.
    """
    def __init__(self, env: ResearchEnv, alpha_sr: float = 0.1, alpha_w: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        super().__init__(env, gamma)
        self.alpha_sr = alpha_sr
        self.alpha_w = alpha_w
        self.epsilon = epsilon
        # successor features: shape (n_states, n_actions, n_states)
        self.M = np.zeros((self.n_states, self.n_actions, self.n_states))
        # weight vector
        self.w = np.zeros(self.n_states)
    def act(self, state: int, exploit: bool = False) -> int:
        # epsilon‑greedy on Q(s,a) = M(s,a)·w
        if not exploit and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        Q = np.einsum('an,n->a', self.M[state], self.w)
        return int(np.argmax(Q))
    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0
        while not done and t < max_steps:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            # compute next action for TD target
            next_action = self.act(next_state, exploit=True)
            # SR update: δ_sr = φ(s) + γ M(s', a') − M(s,a)
            phi_s = np.zeros(self.n_states)
            phi_s[state] = 1.0
            delta_sr = phi_s + self.gamma * self.M[next_state, next_action] - self.M[state, action]
            self.M[state, action] += self.alpha_sr * delta_sr
            # Reward weight update: w ← w + α_w (reward - φ(s)·w) φ(s)
            # Here reward is scalar; φ(s) is indicator vector
            reward_pred = phi_s.dot(self.w)
            error_w = reward - reward_pred
            self.w += self.alpha_w * error_w * phi_s
            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1
        return EpisodeStats(total_return, time.time() - start_time)


class BAMDPSRAgent(BaseAgent):
    """Adaptive SR agent with simple controller and transition model.

    This agent implements a simplified version of the adaptive SR architecture
    described in the project report.  It maintains:

    • Successor features M(s,a) ∈ ℝ^{n_states} updated via TD learning.
    • Reward weight vector w ∈ ℝ^{n_states} updated via linear regression.
    • Transition model T(s,a) predicting the next state.  T is a table of
      counts from which the most likely next state is derived.
    • A controller that monitors the SR error δ_sr and the model error
      δ_model.  The drift score ζ_t = α||δ_sr||² + β||δ_model||² triggers
      exploration when above a threshold.

    When exploring, the agent chooses a random action to gather new data.
    Otherwise it exploits using the policy derived from Q(s,a) = M(s,a)·w.
    The controller resets its drift threshold at the start of each episode.
    """
    def __init__(self, env: ResearchEnv, alpha_sr: float = 0.1, alpha_w: float = 0.1,
                 alpha_t: float = 0.1, gamma: float = 0.95, drift_alpha: float = 1.0,
                 drift_beta: float = 1.0, drift_threshold: float = 0.5,
                 explore_steps: int = 3):
        super().__init__(env, gamma)
        self.alpha_sr = alpha_sr
        self.alpha_w = alpha_w
        self.alpha_t = alpha_t
        self.drift_alpha = drift_alpha
        self.drift_beta = drift_beta
        self.base_threshold = drift_threshold
        self.explore_steps = explore_steps
        # successor features and weight vector
        self.M = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.w = np.zeros(self.n_states)
        # transition model: counts and predicted next state
        # counts[s,a,next_state]
        self.T_counts = np.ones((self.n_states, self.n_actions, self.n_states))  # uniform prior
        # predicted next state (most frequent)
        self.T_pred = np.zeros((self.n_states, self.n_actions), dtype=int)
        self._update_T_pred()
        # internal controller state
        self.drift_score = 0.0
        self.steps_to_explore = 0
    def _update_T_pred(self) -> None:
        self.T_pred = np.argmax(self.T_counts, axis=2)
    def _update_transition(self, s: int, a: int, s_next: int) -> None:
        # update counts
        self.T_counts[s, a, s_next] += 1.0
        # simple decayed counts to avoid unbounded growth
        self.T_counts[s, a] *= (1.0 - self.alpha_t)
        self.T_counts[s, a, s_next] += self.alpha_t
        self.T_pred[s, a] = int(np.argmax(self.T_counts[s, a]))
    def act(self, state: int, exploit: bool = False) -> int:
        # if exploring, choose random action
        if not exploit and self.steps_to_explore > 0:
            self.steps_to_explore -= 1
            return np.random.randint(self.n_actions)
        # else choose greedy action based on Q(s,a)
        Q = np.einsum('an,n->a', self.M[state], self.w)
        return int(np.argmax(Q))
    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0
        # reset drift threshold randomly per episode to avoid repeated exploration
        threshold = self.base_threshold
        self.steps_to_explore = 0
        while not done and t < max_steps:
            # choose action
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            # update transition model
            self._update_transition(state, action, next_state)
            # predicted next state from model
            pred_next = self.T_pred[state, action]
            # SR update
            phi_s = np.zeros(self.n_states)
            phi_s[state] = 1.0
            next_action = self.act(next_state, exploit=True)
            delta_sr = phi_s + self.gamma * self.M[next_state, next_action] - self.M[state, action]
            self.M[state, action] += self.alpha_sr * delta_sr
            # Reward weight update
            reward_pred = phi_s.dot(self.w)
            error_w = reward - reward_pred
            self.w += self.alpha_w * error_w * phi_s
            # Compute model error as 0/1 mismatch between predicted and actual next state
            delta_model = 1.0 if pred_next != next_state else 0.0
            # Compute drift score
            sr_error = np.sum(delta_sr ** 2)
            self.drift_score = self.drift_alpha * sr_error + self.drift_beta * delta_model
            # If drift score exceeds threshold, trigger exploration
            if self.drift_score > threshold:
                self.steps_to_explore = self.explore_steps
                # increase threshold to avoid immediate repeated triggers
                threshold *= 1.5
            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1
        return EpisodeStats(total_return, time.time() - start_time)