"""
NumPy implementation of a Bayes‑Adaptive Successor Representation (SR) agent.

This module provides a torch‑free reference implementation of the alternative
formulation described in the project PDF【338746102192806†L144-L153】.  It models
non‑stationarity via a latent context φ whose influence is summarised by
hand‑crafted history features z_t rather than a recurrent neural encoder.  The
agent maintains a linear successor representation conditioned on state and
latent features, and a linear softmax policy.  Parameter updates are computed
manually using analytic gradients for the SR Bellman residual and the
policy/value objectives.  Although simplified, this implementation preserves
the core idea: use SR losses【338746102192806†L291-L295】 alongside policy and value
losses to adapt to changing dynamics and rewards.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


class NonStationaryMDP:
    """Same as the torch‑based version: see bamdp_sr.py for documentation."""
    def __init__(self, n_states: int, n_actions: int, contexts: List[Dict[str, Any]],
                 initial_state: int = 0, gamma: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.contexts = contexts
        self.initial_state = initial_state
        self.gamma = gamma
        self.current_context = 0
        self.reset()
    def sample_context(self) -> int:
        return np.random.randint(len(self.contexts))
    def reset(self, context: Optional[int] = None) -> int:
        if context is None:
            context = self.sample_context()
        self.current_context = context
        self.state = self.initial_state
        return self.state
    def step(self, action: int) -> Tuple[int, float, bool]:
        trans = self.contexts[self.current_context]['transitions']
        rew = self.contexts[self.current_context]['rewards']
        next_state = trans[self.state][action]
        reward = rew[self.state][action]
        self.state = next_state
        done = False
        return next_state, reward, done


@dataclass
class Transition:
    state: int
    action: int
    reward: float


class BAMDPNumpyAgent:
    """A linear Bayes‑Adaptive SR agent implemented with NumPy.

    The agent uses simple handcrafted history features z_t = [mean of past
    rewards, mean of past state indices] to infer the current context.  It
    maintains separate weight tensors for the successor representation (SR) and
    the policy.  The SR weights have shape (n_actions, input_dim, sr_dim),
    where input_dim = n_states + z_dim and sr_dim = n_states.  The policy
    weights have shape (n_actions, input_dim).  Training updates the weights
    using manual gradient computations combining SR loss, value loss and
    policy loss.
    """
    def __init__(self, env: NonStationaryMDP, z_dim: int = 2, gamma: float = 0.9,
                 lr: float = 0.1, sr_loss_weight: float = 1.0, value_loss_weight: float = 0.5,
                 entropy_weight: float = 0.01):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.z_dim = z_dim
        self.gamma = gamma
        self.lr = lr
        self.sr_loss_weight = sr_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        # Weight initialisation
        self.input_dim = self.n_states + self.z_dim
        # SR weights: (n_actions, input_dim, sr_dim)
        # initialise small random weights
        self.W_sr = np.random.randn(self.n_actions, self.input_dim, self.n_states) * 0.01
        # Policy weights: (n_actions, input_dim)
        self.W_policy = np.random.randn(self.n_actions, self.input_dim) * 0.01

    def _compute_z(self, transitions: List[Transition]) -> np.ndarray:
        """Compute handcrafted latent embedding z from history.

        z[0] = mean of past rewards, z[1] = mean of past state indices.  If
        there is no history, z = zeros.
        """
        if len(transitions) == 0:
            return np.zeros(self.z_dim, dtype=float)
        rewards = [tr.reward for tr in transitions]
        states = [tr.state for tr in transitions]
        return np.array([np.mean(rewards), np.mean(states)], dtype=float)

    def _features(self, state: int, z: np.ndarray) -> np.ndarray:
        """Construct the concatenated feature vector [one_hot(state), z]."""
        one_hot = np.zeros(self.n_states, dtype=float)
        one_hot[state] = 1.0
        return np.concatenate([one_hot, z], axis=0)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax in a numerically stable way."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _sr_predict(self, state: int, z: np.ndarray) -> np.ndarray:
        """Predict successor features for all actions.

        Returns: array of shape (n_actions, n_states) containing M(s,a,z) for each action.
        """
        phi = self._features(state, z)  # (input_dim,)
        # M[a] = phi.T @ W_sr[a]
        # Equivalent to dot phi (input_dim,) with W_sr[a] (input_dim, sr_dim)
        return np.einsum('i,ais->as', phi, self.W_sr)

    def _policy_probs(self, state: int, z: np.ndarray) -> np.ndarray:
        phi = self._features(state, z)
        logits = np.dot(self.W_policy, phi)  # (n_actions,)
        return self._softmax(logits)

    def select_action(self, state: int, z: np.ndarray) -> Tuple[int, float]:
        probs = self._policy_probs(state, z)
        action = np.random.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-12)
        return action, log_prob

    def compute_q_values(self, sr: np.ndarray, reward_weights: np.ndarray) -> np.ndarray:
        # sr shape (n_actions, sr_dim)
        # reward_weights shape (sr_dim,)
        return sr @ reward_weights

    def train_episode(self, max_steps: int = 30) -> float:
        # Sample and reset context
        context_idx = self.env.sample_context()
        state = self.env.reset(context_idx)
        transitions: List[Transition] = []
        log_probs: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        sr_grads = np.zeros_like(self.W_sr)
        policy_grads = np.zeros_like(self.W_policy)
        total_return = 0.0
        discount = 1.0
        # Precompute reward weights: immediate reward for each state in this context
        reward_weights = np.array([max(self.env.contexts[context_idx]['rewards'][s].values())
                                   for s in range(self.n_states)], dtype=float)
        for step in range(max_steps):
            z = self._compute_z(transitions)
            action, log_prob = self.select_action(state, z)
            next_state, reward, done = self.env.step(action)
            # Record
            transitions.append(Transition(state, action, reward))
            log_probs.append(log_prob)
            rewards.append(reward)
            total_return += discount * reward
            discount *= self.gamma
            # Compute SR predictions and Q value for selected action
            sr_pred = self._sr_predict(state, z)  # (n_actions, n_states)
            Q_values = self.compute_q_values(sr_pred, reward_weights)  # (n_actions,)
            values.append(Q_values[action])
            # Compute SR Bellman residual for this step
            # Compute next z and next action for δ_sr
            z_next = self._compute_z(transitions)
            next_action, _ = self.select_action(next_state, z_next)
            sr_next = self._sr_predict(next_state, z_next)
            phi_s = np.zeros(self.n_states)
            phi_s[state] = 1.0
            # δ_sr = φ(s) + γ M(s', a', z') − M(s,a,z)
            delta_sr = phi_s + self.gamma * sr_next[next_action] - sr_pred[action]
            # SR gradient: accumulate for current and next action
            phi_full = self._features(state, z)  # (input_dim,)
            phi_next_full = self._features(next_state, z_next)
            # Update gradient for current action
            sr_grads[action] += (-1.0) * np.outer(phi_full, delta_sr)
            # Update gradient for next action
            sr_grads[next_action] += self.gamma * np.outer(phi_next_full, delta_sr)
            state = next_state
            if done:
                break
        # Compute returns and advantages
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        values_arr = np.array(values)
        advantages = returns - values_arr
        # Normalise advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy gradient
        for t, tr in enumerate(transitions):
            z_t = self._compute_z(transitions[:t])
            phi_t = self._features(tr.state, z_t)
            probs = self._policy_probs(tr.state, z_t)
            for a in range(self.n_actions):
                grad_coeff = advantages[t] * (probs[a] - (1 if a == tr.action else 0))
                policy_grads[a] += grad_coeff * phi_t
            # Entropy regularisation gradient
            if self.entropy_weight > 0:
                for a in range(self.n_actions):
                    policy_grads[a] += (-self.entropy_weight) * (
                        -np.log(probs[a] + 1e-12) - 1.0) * probs[a] * phi_t
        # Value loss gradient: propagate into SR weights
        for t, tr in enumerate(transitions):
            z_t = self._compute_z(transitions[:t])
            phi_t = self._features(tr.state, z_t)
            a = tr.action
            # d/dW_sr[a] (value_loss) = 2 (Q(s,a,z) - G) * φ(s,z) ⊗ reward_weights
            e_v = values_arr[t] - returns[t]
            sr_grads[a] += 2 * e_v * np.outer(phi_t, reward_weights)
        # Combine gradients with weights
        # Total SR gradient = sr_loss_weight * sr_grads + value_loss_weight * sr_grads (from value loss)
        total_sr_grad = self.sr_loss_weight * sr_grads + self.value_loss_weight * sr_grads
        # Parameter update
        self.W_sr -= self.lr * total_sr_grad
        self.W_policy -= self.lr * policy_grads
        return total_return

    def evaluate(self, n_episodes: int = 20, max_steps: int = 30) -> float:
        total = 0.0
        for _ in range(n_episodes):
            context_idx = self.env.sample_context()
            state = self.env.reset(context_idx)
            transitions: List[Transition] = []
            episode_return = 0.0
            discount = 1.0
            for step in range(max_steps):
                z = self._compute_z(transitions)
                # Greedy action from Q values
                sr_pred = self._sr_predict(state, z)
                reward_weights = np.array([
                    max(self.env.contexts[context_idx]['rewards'][s].values())
                    for s in range(self.n_states)
                ])
                Q_vals = self.compute_q_values(sr_pred, reward_weights)
                action = int(np.argmax(Q_vals))
                next_state, reward, done = self.env.step(action)
                transitions.append(Transition(state, action, reward))
                episode_return += discount * reward
                discount *= self.gamma
                state = next_state
                if done:
                    break
            total += episode_return
        return total / n_episodes
