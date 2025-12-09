from __future__ import annotations

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import time

from updated_research_env import ResearchEnv


@dataclass
class EpisodeStats:
    """Simple container for episode statistics."""
    returns: float
    duration: float


class BaseAgent:
    """Base class for reinforcement learning agents."""

    def __init__(self, env: ResearchEnv, gamma: float = 0.95):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = gamma
        # last_trace will store per-step info for the latest episode
        self.last_trace: List[Dict] = []

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
    """Tabular Q-learning agent."""

    def __init__(self, env: ResearchEnv, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.95):
        super().__init__(env, gamma)
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((self.n_states, self.n_actions))

    def act(self, state: int, exploit: bool = False) -> int:
        if not exploit and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        self.last_trace = []
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0

        while not done and t < max_steps:
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            best_next = np.max(self.Q[next_state])
            td_error = reward + self.gamma * best_next - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

            self.last_trace.append({
                "t": t,
                "state": int(state),
                "next_state": int(next_state),
                "action": int(action),
                "reward": float(reward),
                "context": int(info.get("context", -1)),
                "zeta": 0.0,
                "sr_error": 0.0,
                "model_error": 0.0,
                "explore": bool(np.random.random() < self.epsilon),
            })

            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1

        return EpisodeStats(total_return, time.time() - start_time)


class SFAgent(BaseAgent):
    """
    Successor-feature agent with linear reward model.

    Maintains successor features M(s,a) in R^{n_states} and weight vector
    w in R^{n_states}. Q(s,a) = M(s,a) dot w. Learns M via TD learning and
    w via simple regression. Does not explicitly adapt to context drift.
    """

    def __init__(
        self,
        env: ResearchEnv,
        alpha_sr: float = 0.1,
        alpha_w: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        super().__init__(env, gamma)
        self.alpha_sr = alpha_sr
        self.alpha_w = alpha_w
        self.epsilon = epsilon
        self.M = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.w = np.zeros(self.n_states)

    def act(self, state: int, exploit: bool = False) -> int:
        if not exploit and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        Q = np.einsum("an,n->a", self.M[state], self.w)
        return int(np.argmax(Q))

    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        self.last_trace = []
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0

        while not done and t < max_steps:
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)

            next_action = self.act(next_state, exploit=True)
            phi_s = np.zeros(self.n_states)
            phi_s[state] = 1.0

            # SR Bellman update
            delta_sr = phi_s + self.gamma * self.M[next_state, next_action] - self.M[state, action]
            self.M[state, action] += self.alpha_sr * delta_sr

            # Reward weights update
            reward_pred = phi_s.dot(self.w)
            error_w = reward - reward_pred
            self.w += self.alpha_w * error_w * phi_s

            sr_error = float(np.sum(delta_sr ** 2))

            self.last_trace.append({
                "t": t,
                "state": int(state),
                "next_state": int(next_state),
                "action": int(action),
                "reward": float(reward),
                "context": int(info.get("context", -1)),
                "zeta": sr_error,
                "sr_error": sr_error,
                "model_error": 0.0,
                "explore": bool(np.random.random() < self.epsilon),
            })

            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1

        return EpisodeStats(total_return, time.time() - start_time)


class BAMDPSRAgent(BaseAgent):
    """
    Adaptive SR agent with simple controller and transition model.

    Maintains successor features M(s,a), reward weights w, and a tabular
    transition model T_counts(s,a,s'). A drift score

        zeta_t = alpha * ||delta_SR||^2 + beta * (model mismatch)^2

    is computed from the SR Bellman residual and model error. When zeta_t
    exceeds a threshold, the controller triggers a short exploration
    phase (random actions) before returning to exploitation.
    """

    def __init__(
        self,
        env: ResearchEnv,
        alpha_sr: float = 0.1,
        alpha_w: float = 0.1,
        alpha_t: float = 0.1,
        gamma: float = 0.95,
        drift_alpha: float = 1.0,
        drift_beta: float = 1.0,
        drift_threshold: float = 0.5,
        explore_steps: int = 3,
    ):
        super().__init__(env, gamma)
        self.alpha_sr = alpha_sr
        self.alpha_w = alpha_w
        self.alpha_t = alpha_t
        self.drift_alpha = drift_alpha
        self.drift_beta = drift_beta
        self.base_threshold = drift_threshold
        self.explore_steps = explore_steps

        self.M = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.w = np.zeros(self.n_states)
        self.T_counts = np.ones((self.n_states, self.n_actions, self.n_states))
        self.T_pred = np.zeros((self.n_states, self.n_actions), dtype=int)
        self._update_T_pred()

        self.drift_score = 0.0
        self.steps_to_explore = 0

    def _update_T_pred(self) -> None:
        self.T_pred = np.argmax(self.T_counts, axis=2)

    def _update_transition(self, s: int, a: int, s_next: int) -> None:
        # simple exponential moving average on counts
        self.T_counts[s, a, s_next] += 1.0
        self.T_counts[s, a] *= (1.0 - self.alpha_t)
        self.T_counts[s, a, s_next] += self.alpha_t
        self.T_pred[s, a] = int(np.argmax(self.T_counts[s, a]))

    def act(self, state: int, exploit: bool = False) -> int:
        if not exploit and self.steps_to_explore > 0:
            return int(np.random.randint(self.n_actions))
        Q = np.einsum("an,n->a", self.M[state], self.w)
        return int(np.argmax(Q))

    def train_episode(self, max_steps: int = 100) -> EpisodeStats:
        start_time = time.time()
        state = self.env.reset()
        self.last_trace = []
        done = False
        t = 0
        total_return = 0.0
        discount = 1.0

        threshold = self.base_threshold
        self.steps_to_explore = 0

        while not done and t < max_steps:
            exploring = self.steps_to_explore > 0
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)

            # Transition update and model error
            prev_pred = int(self.T_pred[state, action])
            self._update_transition(state, action, next_state)
            pred_next = int(self.T_pred[state, action])

            # SR updates
            phi_s = np.zeros(self.n_states)
            phi_s[state] = 1.0
            next_action = self.act(next_state, exploit=True)
            delta_sr = phi_s + self.gamma * self.M[next_state, next_action] - self.M[state, action]
            self.M[state, action] += self.alpha_sr * delta_sr

            reward_pred = phi_s.dot(self.w)
            error_w = reward - reward_pred
            self.w += self.alpha_w * error_w * phi_s

            sr_error = float(np.sum(delta_sr ** 2))
            model_error = 1.0 if pred_next != next_state else 0.0

            # Drift score
            self.drift_score = self.drift_alpha * sr_error + self.drift_beta * (model_error ** 2)

            # Controller: trigger exploration when drift score is large
            if self.drift_score > threshold:
                self.steps_to_explore = self.explore_steps
                threshold *= 1.5  # hysteresis

            self.last_trace.append({
                "t": t,
                "state": int(state),
                "next_state": int(next_state),
                "action": int(action),
                "reward": float(reward),
                "context": int(info.get("context", -1)),
                "zeta": float(self.drift_score),
                "sr_error": sr_error,
                "model_error": model_error,
                "explore": bool(exploring),
            })

            total_return += discount * reward
            discount *= self.gamma
            state = next_state
            t += 1

        return EpisodeStats(total_return, time.time() - start_time)