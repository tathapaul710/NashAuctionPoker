"""
Deep CFR — replaces regret tables with neural networks.
Two networks per player:
  1. Advantage Net   → approximates counterfactual regrets
  2. Strategy Net    → approximates the average strategy (for Nash output)

Training uses a reservoir buffer to maintain uniform sample distribution
across CFR iterations (as in the original Brown et al. 2019 paper).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Tuple, Optional
from game_engine.games import GameState


# ─────────────────────────────────────────────
#  Reservoir Buffer
#  Maintains a uniform random sample from ALL past data.
#  When full, new items replace old ones with probability n/t.
# ─────────────────────────────────────────────

class ReservoirBuffer:
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.buffer    = []
        self.total_seen = 0

    def add(self, item):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Reservoir sampling: replace with probability capacity/total_seen
            idx = np.random.randint(0, self.total_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size: int) -> List:
        n = min(batch_size, len(self.buffer))
        return [self.buffer[i] for i in np.random.choice(len(self.buffer), n, replace=False)]

    def __len__(self): return len(self.buffer)


# ─────────────────────────────────────────────
#  Neural Networks
# ─────────────────────────────────────────────

class AdvantageNet(nn.Module):
    """
    Maps info-set encoding → advantage (regret) per action.
    Output can be negative (negative regret = this action was worse).
    """
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyNet(nn.Module):
    """
    Maps info-set encoding → probability distribution over actions.
    Output: softmax over actions (valid probability simplex).
    """
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


# ─────────────────────────────────────────────
#  Deep CFR Solver
# ─────────────────────────────────────────────

class DeepCFR:
    def __init__(
        self,
        game_factory,
        state_dim: int,
        num_actions: int,
        num_players: int,
        buffer_capacity: int = 2_000_000,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        train_batch: int = 512,
        train_steps: int = 300,
        device: str = "cpu"
    ):
        self.game_factory  = game_factory
        self.state_dim     = state_dim
        self.num_actions   = num_actions
        self.num_players   = num_players
        self.train_batch   = train_batch
        self.train_steps   = train_steps
        self.device        = torch.device(device)
        self.iterations    = 0
        self.training_log  = []

        # One advantage network + buffer per player
        self.adv_nets = nn.ModuleList([
            AdvantageNet(state_dim, num_actions, hidden_dim).to(self.device)
            for _ in range(num_players)
        ])
        self.adv_buffers = [ReservoirBuffer(buffer_capacity) for _ in range(num_players)]

        # One shared strategy network + buffer (stores avg strategy samples)
        self.strat_net    = StrategyNet(state_dim, num_actions, hidden_dim).to(self.device)
        self.strat_buffers = [ReservoirBuffer(buffer_capacity) for _ in range(num_players)]

        # Optimizers
        self.adv_optims = [
            torch.optim.Adam(net.parameters(), lr=lr)
            for net in self.adv_nets
        ]
        self.strat_optim = torch.optim.Adam(self.strat_net.parameters(), lr=lr)

    # ── Strategy from advantage network (regret matching) ─────────

    @torch.no_grad()
    def get_strategy(self, info_tensor: np.ndarray, player: int, actions: List[int]) -> Dict[int, float]:
        x   = torch.tensor(info_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        adv = self.adv_nets[player](x).squeeze(0).cpu().numpy()
        # Regret matching+: only positive advantages count
        pos = np.maximum(adv[:len(actions)], 0.0)
        total = pos.sum()
        if total > 1e-8:
            probs = pos / total
        else:
            probs = np.ones(len(actions)) / len(actions)
        return {a: float(probs[i]) for i, a in enumerate(actions)}

    # ── CFR traversal with external sampling ──────────────────────

    def _traverse(self, state: GameState, player: int, reach: List[float], t: int) -> List[float]:
        """
        External sampling MCCFR:
        - For the traversing player: explore ALL actions, collect regrets
        - For other players: sample ONE action according to current strategy
        """
        if state.is_terminal():
            return state.returns()

        curr_player = state.current_player()
        actions     = state.legal_actions()
        n           = state.num_players()
        info_tensor = state.info_set_tensor(curr_player)
        info_key    = state.info_set_key(curr_player)
        strategy    = self.get_strategy(info_tensor, curr_player, actions)

        if curr_player == player:
            # Traverse ALL actions for this player (update regrets)
            action_values: Dict[int, List[float]] = {}
            node_value = [0.0] * n

            for a in actions:
                new_reach  = reach.copy()
                new_reach[curr_player] *= strategy[a]
                action_values[a] = self._traverse(state.apply_action(a), player, new_reach, t)
                for i in range(n):
                    node_value[i] += strategy[a] * action_values[a][i]

            # Counterfactual reach (product of all opponents' reaches)
            cf_reach = 1.0
            for i in range(n):
                if i != curr_player:
                    cf_reach *= reach[i]

            # Store (info_tensor, regrets) in advantage buffer
            regrets = np.zeros(self.num_actions, dtype=np.float32)
            for i, a in enumerate(actions):
                regrets[i] = cf_reach * (action_values[a][curr_player] - node_value[curr_player])
            self.adv_buffers[player].add((info_tensor, regrets, t))

            # Store strategy sample for strategy network
            strat_vec = np.zeros(self.num_actions, dtype=np.float32)
            for i, a in enumerate(actions):
                strat_vec[i] = strategy[a]
            self.strat_buffers[player].add((info_tensor, strat_vec, t))

            return node_value

        else:
            # Sample ONE action for non-traversing players
            probs = np.array([strategy[a] for a in actions], dtype=float)

            # ---- FIX START ----
            probs = np.maximum(probs, 0)  # remove negative values

            total = probs.sum()
            if total <= 1e-8:
                probs = np.ones(len(probs)) / len(probs)  # fallback uniform
            else:
                probs = probs / total  # normalize
            # ---- FIX END ----
            chosen = actions[np.random.choice(len(actions), p=probs)]
            new_reach = reach.copy()
            new_reach[curr_player] *= strategy[chosen]
            return self._traverse(state.apply_action(chosen), player, new_reach, t)

    # ── Train advantage network ───────────────────────────────────

    def _train_advantage_net(self, player: int) -> float:
        buf = self.adv_buffers[player]
        if len(buf) < self.train_batch:
            return 0.0

        net   = self.adv_nets[player]
        optim = self.adv_optims[player]
        net.train()
        total_loss = 0.0

        for _ in range(self.train_steps):
            batch = buf.sample(self.train_batch)
            states  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(self.device)
            targets = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).to(self.device)
            weights = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)

            pred = net(states)
            # Weighted MSE (later iterations get higher weight)
            loss = (weights.unsqueeze(1) * (pred - targets) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()

        return total_loss / self.train_steps

    # ── Train strategy network ────────────────────────────────────

    def _train_strategy_net(self) -> float:
        all_samples = []
        for buf in self.strat_buffers:
            if len(buf) >= self.train_batch // self.num_players:
                all_samples.extend(buf.sample(self.train_batch // self.num_players))

        if not all_samples:
            return 0.0

        self.strat_net.train()
        total_loss = 0.0

        for _ in range(self.train_steps // 2):
            batch_size = min(self.train_batch, len(all_samples))
            idx     = np.random.choice(len(all_samples), batch_size, replace=False)
            batch   = [all_samples[i] for i in idx]
            states  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(self.device)
            targets = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).to(self.device)
            weights = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)

            pred = self.strat_net(states)
            loss = (weights.unsqueeze(1) * (pred - targets) ** 2).mean()
            self.strat_optim.zero_grad()
            loss.backward()
            self.strat_optim.step()
            total_loss += loss.item()

        return total_loss / (self.train_steps // 2)

    # ── Main training loop ────────────────────────────────────────

    def train(self, num_iterations: int, traversals_per_iter: int = 100, seed: int = 42):
        rng = np.random.default_rng(seed)
        log = []

        for t in range(1, num_iterations + 1):
            # Each iteration: traverse for each player
            for player in range(self.num_players):
                for _ in range(traversals_per_iter):
                    game_seed = int(rng.integers(0, 100000))
                    state = self.game_factory(seed=game_seed)
                    self._traverse(state, player, [1.0] * self.num_players, t)

            # Train networks every iteration
            adv_losses = [self._train_advantage_net(p) for p in range(self.num_players)]
            strat_loss = self._train_strategy_net()

            self.iterations += 1
            entry = {
                "iteration": t,
                "adv_losses": adv_losses,
                "strat_loss": strat_loss,
                "buffer_sizes": [len(buf) for buf in self.adv_buffers],
            }
            log.append(entry)

            if t % max(1, num_iterations // 10) == 0:
                print(f"  Iter {t:4d} | Adv losses: {[f'{l:.4f}' for l in adv_losses]} | "
                      f"Strat loss: {strat_loss:.4f} | "
                      f"Buffer: {[len(b) for b in self.adv_buffers]}")

        self.training_log = log
        return log

    # ── Policy inference ──────────────────────────────────────────

    @torch.no_grad()
    def get_policy(self, state: GameState, player: int) -> Dict[int, float]:
        """Get the Nash (average) strategy at a given info set."""
        if state.is_terminal(): return {}
        if state.current_player() != player: return {}
        actions     = state.legal_actions()
        info_tensor = torch.tensor(
            state.info_set_tensor(player), dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        self.strat_net.eval()
        probs = self.strat_net(info_tensor).squeeze(0).cpu().numpy()
        probs = np.maximum(probs, 0)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
        return {a: float(probs[i]) for i, a in enumerate(actions)}

    # ── Model persistence ─────────────────────────────────────────

    def save_model(self, path: str):
        """Save the strategy network parameters for later inference."""
        torch.save(self.strat_net.state_dict(), path)

    def load_model(self, path: str):
        """Load the strategy network parameters from disk."""
        self.strat_net.load_state_dict(torch.load(path, map_location=self.device))
        self.strat_net.eval()
