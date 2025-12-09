from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class NonStationaryMDP:
    """
    Simple tabular MDP with multiple contexts.

    Each context defines deterministic transitions and rewards over
    discrete states s in {0,...,n_states-1}, actions a in {0,...,n_actions-1}.
    Context is fixed within an episode but varies across episodes.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        contexts: List[Dict[str, Any]],
        initial_state: int = 0,
        gamma: float = 0.95,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.contexts = contexts
        self.initial_state = initial_state
        self.gamma = gamma
        self.current_context = 0
        self.state = initial_state

    def sample_context(self) -> int:
        return int(np.random.randint(len(self.contexts)))

    def reset(self, context: Optional[int] = None) -> int:
        if context is None:
            context = self.sample_context()
        self.current_context = context
        self.state = self.initial_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        trans = self.contexts[self.current_context]["transitions"]
        rew = self.contexts[self.current_context]["rewards"]
        next_state = int(trans[self.state][action])
        reward = float(rew[self.state][action])
        self.state = next_state
        done = False
        return next_state, reward, done


class HistoryEncoder(nn.Module):
    """
    GRU-based encoder mapping a sequence of (state, action, reward)
    triples into a latent embedding z_t.
    """

    def __init__(self, n_states: int, n_actions: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.input_dim = n_states + n_actions + 1
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(history)
        h = h.squeeze(0)
        z = self.fc(h)
        return z


class SRNetwork(nn.Module):
    """
    Successor representation network conditioned on state and latent z.
    """

    def __init__(self, n_states: int, n_actions: int, latent_dim: int,
                 hidden_dim: int = 128, sr_dim: Optional[int] = None):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.sr_dim = sr_dim or n_states
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions * self.sr_dim)

    def forward(self, state_idx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s_embed = self.state_embed(state_idx)
        h = torch.cat([s_embed, z], dim=-1)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        sr = out.view(-1, self.n_actions, self.sr_dim)
        return sr


class PolicyNetwork(nn.Module):
    """
    Policy network mapping (state, z) to logits over actions.
    """

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


class TransitionModel(nn.Module):
    """
    Simple learned transition model T_eta(s, a, z) predicting next-state distribution.
    """

    def __init__(self, n_states: int, n_actions: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_embed = nn.Embedding(n_actions, hidden_dim)
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_states)

    def forward(self, state_idx: torch.Tensor, action_idx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_embed(state_idx)
        a_emb = self.action_embed(action_idx)
        h = torch.cat([s_emb, a_emb, z], dim=-1)
        h = F.relu(self.fc1(h))
        logits = self.fc2(h)
        return logits


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int


class BayesAdaptiveSRAgent:
    """
    Bayes-adaptive SR agent with GRU-based history encoder and learned SR and transition models.

    The agent maintains:
        - History encoder producing latent z_t from past trajectory.
        - SR network psi_theta(s, z) giving successor features for each action.
        - Policy network pi_omega(a | s, z).
        - Transition model T_eta(s, a, z) predicting next state distribution.

    At training time we compute:
        - Policy loss (advantage-weighted log-prob).
        - Value loss from Q-values derived from SR + reward vector.
        - SR loss enforcing the SR Bellman equation.
        - Transition loss as cross-entropy between predicted and actual next state.

    A drift-like signal zeta_t is logged per step as a combination of SR error
    and transition-model error so it can be compared to simpler tabular controllers.
    """

    def __init__(
        self,
        env: NonStationaryMDP,
        n_states: int,
        n_actions: int,
        latent_dim: int = 16,
        gamma: float = 0.95,
        sr_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        entropy_weight: float = 0.01,
        trans_loss_weight: float = 1.0,
        zeta_alpha: float = 1.0,
        zeta_beta: float = 1.0,
    ):
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.sr_loss_weight = sr_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.trans_loss_weight = trans_loss_weight
        self.zeta_alpha = zeta_alpha
        self.zeta_beta = zeta_beta

        self.encoder = HistoryEncoder(n_states, n_actions, latent_dim)
        self.sr_network = SRNetwork(n_states, n_actions, latent_dim)
        self.policy_network = PolicyNetwork(n_states, n_actions, latent_dim)
        self.transition_model = TransitionModel(n_states, n_actions, latent_dim)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.sr_network.parameters())
            + list(self.policy_network.parameters())
            + list(self.transition_model.parameters()),
            lr=1e-3,
        )

        self.last_trace: List[Dict[str, Any]] = []

    def encode_history(self, transitions: List[Transition]) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        seq_len = len(transitions)
        hist = torch.zeros(1, seq_len, self.n_states + self.n_actions + 1, device=device)
        for i, tr in enumerate(transitions):
            hist[0, i, tr.state] = 1.0
            hist[0, i, self.n_states + tr.action] = 1.0
            hist[0, i, -1] = tr.reward
        z = self.encoder(hist)
        return z

    def compute_q_values(self, sr: torch.Tensor, reward_vec: torch.Tensor) -> torch.Tensor:
        """
        sr: (batch, n_actions, sr_dim), reward_vec: (batch, sr_dim)
        returns: (batch, n_actions)
        """
        Q = torch.einsum("ban,bn->ba", sr, reward_vec)
        return Q

    def run_episode(self, max_steps: int = 100) -> float:
        """
        Run one training episode and update parameters. Logs full trajectory
        with per-step zeta_t, SR error and transition-model error.
        """
        device = next(self.encoder.parameters()).device

        context_idx = self.env.sample_context()
        state = self.env.reset(context_idx)

        transitions: List[Transition] = []
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        values: List[torch.Tensor] = []
        trans_logits_list: List[torch.Tensor] = []
        next_state_targets: List[int] = []

        self.last_trace = []

        for t in range(max_steps):
            if transitions:
                z = self.encode_history(transitions)
            else:
                z = torch.zeros((1, self.latent_dim), device=device)

            sr_pred = self.sr_network(torch.tensor([state], device=device), z)
            reward_vec = torch.tensor(
                [max(self.env.contexts[context_idx]["rewards"][s]) for s in range(self.n_states)],
                dtype=torch.float,
                device=device,
            ).unsqueeze(0)
            Q = self.compute_q_values(sr_pred, reward_vec)

            logits = self.policy_network(torch.tensor([state], device=device), z)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = torch.max(Q, dim=-1).values

            next_state, reward, done = self.env.step(int(action.item()))

            trans_logits = self.transition_model(
                torch.tensor([state], device=device),
                action.unsqueeze(0),
                z,
            )

            transitions.append(Transition(state=state, action=int(action.item()), reward=float(reward), next_state=next_state))
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            trans_logits_list.append(trans_logits)
            next_state_targets.append(next_state)

            self.last_trace.append({
                "t": t,
                "state": int(state),
                "next_state": int(next_state),
                "action": int(action.item()),
                "reward": float(reward),
                "context": int(context_idx),
                "zeta": None,
                "sr_error": None,
                "model_error": None,
                "explore": False,
            })

            state = next_state
            if done:
                break

        T = len(rewards)
        if T == 0:
            return 0.0

        returns = torch.zeros(T, device=device)
        G = 0.0
        for i in reversed(range(T)):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        values_tensor = torch.stack(values).squeeze(-1)
        advantages = returns - values_tensor.detach()

        log_probs_tensor = torch.stack(log_probs)
        policy_loss = -(log_probs_tensor * advantages).sum()

        sr_loss = torch.tensor(0.0, device=device)
        sr_errors: List[float] = []

        for i, tr in enumerate(transitions):
            if i > 0:
                z_i = self.encode_history(transitions[:i])
            else:
                z_i = torch.zeros((1, self.latent_dim), device=device)

            sr_i = self.sr_network(torch.tensor([tr.state], device=device), z_i)
            sr_sa = sr_i[0, tr.action]

            phi_s = torch.zeros(self.n_states, device=device)
            phi_s[tr.state] = 1.0

            if i < T - 1:
                next_tr = transitions[i + 1]
                if i + 1 > 0:
                    z_next = self.encode_history(transitions[: i + 1])
                else:
                    z_next = torch.zeros((1, self.latent_dim), device=device)
                sr_next_all = self.sr_network(torch.tensor([next_tr.state], device=device), z_next)
                sr_next = sr_next_all[0, next_tr.action]
                target = phi_s + self.gamma * sr_next
            else:
                target = phi_s

            delta_sr = target - sr_sa
            sr_step_loss = (delta_sr ** 2).sum()
            sr_loss = sr_loss + sr_step_loss

            sr_err_val = float(sr_step_loss.detach().cpu().item())
            sr_errors.append(sr_err_val)

        value_loss = ((values_tensor - returns) ** 2).sum()

        trans_loss = torch.tensor(0.0, device=device)
        model_errors: List[float] = []
        for logits, ns in zip(trans_logits_list, next_state_targets):
            target = torch.tensor([ns], dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target)
            trans_loss = trans_loss + ce
            model_errors.append(float(ce.detach().cpu().item()))

        entropies = []
        for i, tr in enumerate(transitions):
            if i > 0:
                z_i = self.encode_history(transitions[:i])
            else:
                z_i = torch.zeros((1, self.latent_dim), device=device)
            logits_i = self.policy_network(torch.tensor([tr.state], device=device), z_i)
            dist_i = Categorical(logits=logits_i)
            entropies.append(dist_i.entropy())
        entropy_loss = -self.entropy_weight * torch.stack(entropies).sum() if entropies else 0.0

        total_loss = (
            policy_loss
            + self.value_loss_weight * value_loss
            + self.sr_loss_weight * sr_loss
            + self.trans_loss_weight * trans_loss
            + entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for i in range(T):
            sr_err = sr_errors[i]
            model_err = model_errors[i]
            zeta = self.zeta_alpha * sr_err + self.zeta_beta * model_err
            self.last_trace[i]["sr_error"] = sr_err
            self.last_trace[i]["model_error"] = model_err
            self.last_trace[i]["zeta"] = zeta

        return float(returns[0].detach().cpu().item())

    def evaluate(self, n_episodes: int = 10, max_steps: int = 100) -> float:
        device = next(self.encoder.parameters()).device
        total = 0.0
        for _ in range(n_episodes):
            context_idx = self.env.sample_context()
            state = self.env.reset(context_idx)
            transitions: List[Transition] = []
            discount = 1.0
            episode_return = 0.0

            for t in range(max_steps):
                if transitions:
                    z = self.encode_history(transitions)
                else:
                    z = torch.zeros((1, self.latent_dim), device=device)

                sr_pred = self.sr_network(torch.tensor([state], device=device), z)
                reward_vec = torch.tensor(
                    [max(self.env.contexts[context_idx]["rewards"][s]) for s in range(self.n_states)],
                    dtype=torch.float,
                    device=device,
                ).unsqueeze(0)
                Q = self.compute_q_values(sr_pred, reward_vec)
                action = int(torch.argmax(Q[0]).item())

                next_state, reward, done = self.env.step(action)
                transitions.append(Transition(state, action, reward, next_state))

                episode_return += discount * reward
                discount *= self.gamma
                state = next_state
                if done:
                    break

            total += episode_return
        return total / n_episodes