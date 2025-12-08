"""
Implementation of a Bayes‑Adaptive Successor Representation (SR) agent for
non‑stationary environments.  The model follows the alternative formulation
described in the project PDF: non‑stationarity is modeled as a latent
context φ that evolves over time according to a prior p(φ) and a Markov
transition p(φ_t|φ_{t−1})【338746102192806†L140-L150】.  Rather than
maintaining an exact belief over this latent variable, we amortize inference
using an encoder fψ that maps a history of transitions to a low‑dimensional
latent embedding z_t【338746102192806†L144-L153】.  The SR network ψθ and the
policy πω both condition on this embedding to produce context‑dependent
successor features and control signals【338746102192806†L162-L165】.  At
training time the agent samples a context φ from the prior, rolls out an
episode in the corresponding MDP M_φ and jointly optimizes the encoder,
SR and policy networks to maximise expected return subject to the
successor‑representation Bellman constraints【338746102192806†L170-L181】.

This implementation is intentionally minimal and educational: it uses a
simple tabular environment with two latent contexts, a recurrent encoder
based on a Gated Recurrent Unit (GRU) to produce latent embeddings, and
separate multilayer perceptrons for the SR network and policy.  A value
function is derived from the successor features by taking their inner
product with a known reward weight vector w_φ.  The total loss at each
time–step includes:

  • A policy loss (actor) based on the REINFORCE/advantage formulation.
  • A value loss (critic) that regresses the predicted Q‑value towards a
    temporal difference target.
  • An SR loss that enforces the SR Bellman equation by minimising the
    squared residual δ_SR_t = φ(s_t) + γ ψθ(s_{t+1}, a_{t+1}, z_{t+1}) − ψθ(s_t, a_t, z_t)【338746102192806†L291-L295】.

Because the encoder conditions on the entire history (state, action,
reward), the agent is able to infer changes in the latent context and
adapt both its SR and control policy accordingly.  See `test_bamdp_sr.py`
for a simple demonstration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


class NonStationaryMDP:
    """A simple tabular MDP with multiple contexts (tasks).

    Each context φ defines its own transition and reward dynamics.  When a
    context is selected, it remains fixed for the duration of an episode.
    This environment does not change the context within an episode; however
    training across episodes exposes the agent to different contexts drawn
    from the prior.  The state space is {0, 1, …, n_states−1}, actions are
    integers in {0, …, n_actions−1}.  Transitions are deterministic given
    context for simplicity.
    """

    def __init__(self, n_states: int, n_actions: int, contexts: List[Dict[str, Any]],
                 initial_state: int = 0, gamma: float = 0.9):
        """Create an environment.

        Args:
            n_states: number of discrete states.
            n_actions: number of discrete actions.
            contexts: list of dictionaries, each specifying a context with
                      keys 'transitions' and 'rewards'.  transitions[s][a]
                      returns next state, rewards[s][a] returns reward.
            initial_state: starting state for each episode.
            gamma: discount factor (used by the agent; the environment
                   doesn't enforce discounting).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.contexts = contexts
        self.initial_state = initial_state
        self.gamma = gamma
        # validate contexts
        for c in contexts:
            assert len(c['transitions']) == n_states
            assert len(c['rewards']) == n_states
        self.current_context = 0
        self.reset()

    def sample_context(self) -> int:
        """Sample a context index uniformly from available contexts."""
        return np.random.randint(len(self.contexts))

    def reset(self, context: Optional[int] = None) -> int:
        """Reset environment at the beginning of an episode.

        Optionally specify a context index; otherwise one is sampled
        uniformly.  Returns the initial state.
        """
        if context is None:
            context = self.sample_context()
        self.current_context = context
        self.state = self.initial_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take a step in the environment given an action.

        Args:
            action: integer in {0, …, n_actions−1}.
        Returns:
            next_state, reward, done.  `done` is True at terminal states
            (here: always False for simplicity).  For finite tasks one
            could define absorbing end states.
        """
        trans = self.contexts[self.current_context]['transitions']
        rew = self.contexts[self.current_context]['rewards']
        next_state = trans[self.state][action]
        reward = rew[self.state][action]
        # In this toy environment we never reach a terminal state
        self.state = next_state
        done = False
        return next_state, reward, done


class HistoryEncoder(nn.Module):
    """GRU‑based encoder that maps a sequence of (state, action, reward) triples to a
    latent embedding z_t.

    The encoder takes an input of shape (batch, seq_len, input_dim) where
    input_dim = n_states + n_actions + 1.  States and actions are encoded
    as one‑hot vectors and reward is appended as a scalar.  The final
    hidden state of the GRU is projected through a linear layer to
    produce the latent vector z_t.
    """

    def __init__(self, n_states: int, n_actions: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.input_dim = n_states + n_actions + 1
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode a batch of histories into latent vectors.

        Args:
            history: tensor of shape (batch, seq_len, input_dim).
        Returns:
            latent vectors z of shape (batch, latent_dim).
        """
        # Pass through GRU
        _, h = self.gru(history)
        # h shape: (1, batch, hidden_dim)
        h = h.squeeze(0)
        z = self.fc(h)
        return z


class SRNetwork(nn.Module):
    """Successor representation network conditioned on state and latent context.

    Given a state index and latent embedding z_t, this network outputs a
    tensor of shape (num_actions, sr_dim) representing the predicted
    discounted sum of future state‑feature vectors φ(s') for each action.
    Here φ(s) is chosen to be the one‑hot state vector for simplicity.
    """

    def __init__(self, n_states: int, n_actions: int, latent_dim: int,
                 hidden_dim: int = 128, sr_dim: Optional[int] = None):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        # We set sr_dim to n_states by default, because φ(s) is one‑hot of length n_states.
        self.sr_dim = sr_dim or n_states
        # Embed the discrete state into a continuous vector
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        # Combine state embedding and latent context, then produce SR predictions
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions * self.sr_dim)

    def forward(self, state_idx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute successor representations for each action.

        Args:
            state_idx: tensor of shape (batch,) with integer state indices.
            z: tensor of shape (batch, latent_dim) with latent embeddings.
        Returns:
            sr: tensor of shape (batch, n_actions, sr_dim).
        """
        s_embed = self.state_embed(state_idx)  # (batch, hidden_dim)
        h = torch.cat([s_embed, z], dim=-1)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        sr = out.view(-1, self.n_actions, self.sr_dim)
        return sr


class PolicyNetwork(nn.Module):
    """Policy network mapping state and latent embedding to a categorical action distribution."""
    def __init__(self, n_states: int, n_actions: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state_idx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s_embed = self.state_embed(state_idx)
        h = torch.cat([s_embed, z], dim=-1)
        h = F.relu(self.fc1(h))
        logits = self.fc2(h)
        return logits


@dataclass
class Transition:
    """Container for storing a single transition."""
    state: int
    action: int
    reward: float


class BAMDPAgent:
    """Agent encapsulating encoder, SR and policy networks along with training logic."""

    def __init__(self, env: NonStationaryMDP, latent_dim: int = 8, enc_hidden: int = 64,
                 sr_hidden: int = 128, policy_hidden: int = 128, gamma: float = 0.9,
                 lr: float = 1e-3, sr_loss_weight: float = 1.0, value_loss_weight: float = 0.5,
                 entropy_weight: float = 0.01):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.encoder = HistoryEncoder(self.n_states, self.n_actions, latent_dim, enc_hidden)
        self.sr_network = SRNetwork(self.n_states, self.n_actions, latent_dim, sr_hidden)
        self.policy_network = PolicyNetwork(self.n_states, self.n_actions, latent_dim, policy_hidden)
        # Initialise optimiser over all parameters
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.sr_network.parameters()) +
            list(self.policy_network.parameters()), lr=lr
        )
        # Reward weight vectors for each context; use one‑hot identity by default.
        # Each context may override this when computing Q.  In our simple
        # environment the reward weight vector is the vector of immediate
        # rewards for each state; here we assume φ(s) is one‑hot so w_φ is
        # identical to the reward vector.
        self.sr_loss_weight = sr_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight

    def encode_history(self, transitions: List[Transition]) -> torch.Tensor:
        """Encode the full history of transitions into a latent vector z.

        Args:
            transitions: list of Transition objects recorded so far.
        Returns:
            z: latent embedding vector (1, latent_dim).
        """
        # Build history tensor: each row is [one_hot_state, one_hot_action, reward]
        seq_len = len(transitions)
        device = next(self.encoder.parameters()).device
        history = torch.zeros((1, seq_len, self.n_states + self.n_actions + 1), device=device)
        for t, tr in enumerate(transitions):
            state_one_hot = F.one_hot(torch.tensor(tr.state, device=device), self.n_states).float()
            action_one_hot = F.one_hot(torch.tensor(tr.action, device=device), self.n_actions).float()
            history[0, t, :self.n_states] = state_one_hot
            history[0, t, self.n_states:self.n_states + self.n_actions] = action_one_hot
            history[0, t, -1] = tr.reward
        z = self.encoder(history)
        return z

    def select_action(self, state: int, z: torch.Tensor) -> Tuple[int, float]:
        """Sample an action from the policy and return action and log probability."""
        state_tensor = torch.tensor([state], dtype=torch.long, device=z.device)
        logits = self.policy_network(state_tensor, z)  # (1, n_actions)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_q_values(self, sr: torch.Tensor, reward_weights: torch.Tensor) -> torch.Tensor:
        """Compute Q‑values by taking inner product of successor features with reward weights.

        Args:
            sr: (batch, n_actions, sr_dim) successor features predicted by SR network.
            reward_weights: (batch, sr_dim) weight vectors for each context; here
                            this is the expected immediate reward vector for
                            each state in the context.
        Returns:
            Q: (batch, n_actions) predicted Q‑values.
        """
        # Multiply sr and reward_weights: (batch, n_actions, sr_dim) • (batch, sr_dim) → (batch, n_actions)
        Q = torch.einsum('ban, bn -> ba', sr, reward_weights)
        return Q

    def train_episode(self, max_steps: int = 100) -> float:
        """Run a single episode and update parameters.

        Args:
            max_steps: maximum number of transitions per episode.
        Returns:
            total return obtained in the episode.
        """
        # Sample a new context and reset environment
        context_idx = self.env.sample_context()
        state = self.env.reset(context_idx)
        transitions: List[Transition] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        sr_losses: List[torch.Tensor] = []
        total_return = 0.0
        discount = 1.0
        device = next(self.encoder.parameters()).device
        # Precompute reward weight vector for this context; φ(s) is one‑hot of size n_states
        # so the reward weight vector is the immediate reward for each state.
        reward_vec = torch.tensor([
            max(self.env.contexts[context_idx]['rewards'][s]) for s in range(self.n_states)
        ], dtype=torch.float, device=device)
        reward_vec = reward_vec.unsqueeze(0)  # (1, n_states)
        for step in range(max_steps):
            # Encode history to get latent z
            if len(transitions) > 0:
                z = self.encode_history(transitions)
            else:
                # If no transitions yet, use zero vector
                z = torch.zeros((1, self.latent_dim), dtype=torch.float, device=device)
            # Select action
            action, log_prob = self.select_action(state, z)
            # Step environment
            next_state, reward, done = self.env.step(action)
            # Record transition and training buffers
            transitions.append(Transition(state=state, action=action, reward=reward))
            log_probs.append(log_prob)
            rewards.append(reward)
            total_return += discount * reward
            discount *= self.gamma
            # Value estimation
            # Compute SR predictions and Q values for current state
            sr_current = self.sr_network(torch.tensor([state], device=device), z)  # (1, n_actions, sr_dim)
            Q_current = self.compute_q_values(sr_current, reward_vec)  # (1, n_actions)
            # Append value of the selected action
            values.append(Q_current[0, action])
            # Compute SR Bellman residual δ_SR_t
            # To compute M(s_{t+1}, a_{t+1}), we need z_{t+1} based on history including next transition.
            # We approximate z_{t+1} by adding the next transition to history and re‑encoding.
            next_transitions = transitions.copy()
            next_transitions[-1] = Transition(state=state, action=action, reward=reward)  # ensure copy
            # Compute z' after taking next transition; note this includes next_state which becomes state in next step
            z_next = self.encode_history(next_transitions)
            # For δ_SR we need M(s_{t+1}, a', z_{t+1}), where a' is the next action sampled from policy under z'
            next_action, _ = self.select_action(next_state, z_next)
            sr_next = self.sr_network(torch.tensor([next_state], device=device), z_next)  # (1, n_actions, sr_dim)
            # δ_SR_t = φ(s) + γ M(s', a', z') − M(s, a, z)
            phi_s = F.one_hot(torch.tensor(state, device=device), num_classes=self.n_states).float().unsqueeze(0)  # (1, n_states)
            sr_next_a = sr_next[0, next_action]  # (sr_dim,)
            sr_current_a = sr_current[0, action]  # (sr_dim,)
            delta_sr = phi_s - sr_current_a + self.gamma * sr_next_a
            sr_losses.append((delta_sr.pow(2).sum()) / 2.0)
            # Transition
            state = next_state
            if done:
                break
        # After collecting episode, compute returns and update
        # Compute advantages and losses
        R = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        returns = []
        # Compute returns backwards
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)
        values_tensor = torch.stack(values)  # (T,)
        log_probs_tensor = torch.stack(log_probs)  # (T,)
        # Normalise returns for stability
        if len(returns_tensor) > 1:
            returns_norm = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        else:
            returns_norm = returns_tensor
        advantages = returns_norm - values_tensor.detach()
        # Policy loss
        policy_loss = -(log_probs_tensor * advantages).sum()
        # Value (critic) loss: mean squared error between returns and predicted values
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        # SR loss
        sr_loss = torch.stack(sr_losses).sum()
        # Entropy loss for exploration
        # Encourage policy entropy across all time–steps; compute distribution and entropy
        entropies = []
        for i, (trans, log_prob) in enumerate(zip(transitions, log_probs)):
            # reconstruct z for each step
            z_step = self.encode_history(transitions[:i]) if i > 0 else torch.zeros((1, self.latent_dim), device=device)
            logits = self.policy_network(torch.tensor([trans.state], device=device), z_step)
            dist = Categorical(logits=logits)
            entropies.append(dist.entropy())
        if entropies:
            entropy_loss = -self.entropy_weight * torch.stack(entropies).sum()
        else:
            entropy_loss = 0.0
        # Total loss
        total_loss = policy_loss + self.value_loss_weight * value_loss + self.sr_loss_weight * sr_loss + entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_return

    def evaluate(self, n_episodes: int = 10, max_steps: int = 100) -> float:
        """Run the agent without learning and return the average return."""
        device = next(self.encoder.parameters()).device
        total = 0.0
        for _ in range(n_episodes):
            context_idx = self.env.sample_context()
            state = self.env.reset(context_idx)
            transitions: List[Transition] = []
            episode_return = 0.0
            discount = 1.0
            for step in range(max_steps):
                # Encode history into z
                if transitions:
                    z = self.encode_history(transitions)
                else:
                    z = torch.zeros((1, self.latent_dim), device=device)
                # Greedy action based on Q
                sr_pred = self.sr_network(torch.tensor([state], device=device), z)
                reward_vec = torch.tensor([
                    max(self.env.contexts[context_idx]['rewards'][s]) for s in range(self.n_states)
                ], dtype=torch.float, device=device).unsqueeze(0)
                Q = self.compute_q_values(sr_pred, reward_vec)
                action = int(torch.argmax(Q[0]).item())
                # Step env
                next_state, reward, done = self.env.step(action)
                transitions.append(Transition(state, action, reward))
                episode_return += discount * reward
                discount *= self.gamma
                state = next_state
                if done:
                    break
            total += episode_return
        return total / n_episodes
