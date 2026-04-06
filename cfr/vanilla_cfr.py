"""
Vanilla CFR (Counterfactual Regret Minimization)
Used as baseline before adding neural network approximation.
Works for small games (Leduc, Kuhn) where state table fits in memory.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from game_engine.games import GameState, LeducState


class VanillaCFR:
    def __init__(self, game_factory):
        self.game_factory  = game_factory
        self.regret_sum    = defaultdict(lambda: defaultdict(float))
        self.strategy_sum  = defaultdict(lambda: defaultdict(float))
        self.iterations    = 0

    # ── Core regret-matching ──────────────────────────────────────

    def get_strategy(self, info_key: str, actions: List[int]) -> Dict[int, float]:
        """Regret-matching: convert cumulative regrets → mixed strategy."""
        regrets = self.regret_sum[info_key]
        pos_regrets = {a: max(0.0, regrets[a]) for a in actions}
        total = sum(pos_regrets.values())
        if total > 0:
            return {a: pos_regrets[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}  # uniform

    def get_average_strategy(self, info_key: str, actions: List[int]) -> Dict[int, float]:
        """The Nash strategy is the TIME-AVERAGE of all strategies played."""
        strat = self.strategy_sum[info_key]
        total = sum(strat[a] for a in actions)
        if total > 0:
            return {a: strat[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # ── CFR traversal ─────────────────────────────────────────────

    def cfr(self, state: GameState, reach: List[float]) -> List[float]:
        """
        Recursive CFR traversal.
        reach[i] = probability of reaching this node assuming player i plays
                   to reach it (product of their actions along the path).
        Returns: counterfactual values for each player.
        """
        if state.is_terminal():
            return state.returns()

        player  = state.current_player()
        actions = state.legal_actions()
        info_key = state.info_set_key(player)
        strategy = self.get_strategy(info_key, actions)

        n = state.num_players()

        # Recurse over all actions
        action_values: Dict[int, List[float]] = {}
        node_value = [0.0] * n

        for a in actions:
            new_state = state.apply_action(a)
            new_reach = reach.copy()
            new_reach[player] *= strategy[a]
            action_values[a] = self.cfr(new_state, new_reach)
            for i in range(n):
                node_value[i] += strategy[a] * action_values[a][i]

        # Counterfactual reach: product of all OTHER players' reach probs
        cf_reach = 1.0
        for i in range(n):
            if i != player:
                cf_reach *= reach[i]

        # Update regrets (only for the acting player)
        for a in actions:
            regret = cf_reach * (action_values[a][player] - node_value[player])
            self.regret_sum[info_key][a] += regret

        # Accumulate strategy weighted by reach
        for a in actions:
            self.strategy_sum[info_key][a] += reach[player] * strategy[a]

        return node_value

    # ── Training loop ─────────────────────────────────────────────

    def train(self, num_iterations: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        exploitabilities = []

        for t in range(1, num_iterations + 1):
            state = self.game_factory(seed=int(rng.integers(0, 10000)))
            n = state.num_players()
            self.cfr(state, [1.0] * n)
            self.iterations += 1

            if t % max(1, num_iterations // 20) == 0:
                expl = self.exploitability()
                exploitabilities.append((t, expl))

        return exploitabilities

    # ── Exploitability (Nash convergence metric) ──────────────────

    def exploitability(self) -> float:
        """
        Measures how much a best-response opponent can exploit our avg strategy.
        Lower = closer to Nash Equilibrium. At NE = 0.

        For each player p, compute:
          gap_p = BR_value(p) - avg_strategy_value(p)
        Exploitability = average of max(0, gap_p) across players.

        Note: _avg_strategy_value(state) returns the value for a SINGLE player
        (the current player at the root), so we compute it separately per player.
        """
        state = self.game_factory(seed=0)
        n = state.num_players()
        total_gap = 0.0
        for p in range(n):
            br_val  = self._best_response_value(state, [1.0] * n)[p]
            avg_val = self._avg_strategy_value_for_player(state, p)
            total_gap += max(0.0, br_val - avg_val)
        return total_gap / n

    def _avg_strategy_value_for_player(self, state: GameState, target_player: int) -> float:
        """Expected payoff for target_player when all players follow avg strategy."""
        if state.is_terminal():
            return state.returns()[target_player]
        player   = state.current_player()
        actions  = state.legal_actions()
        info_key = state.info_set_key(player)
        strategy = self.get_average_strategy(info_key, actions)
        total = 0.0
        for a in actions:
            child_val = self._avg_strategy_value_for_player(state.apply_action(a), target_player)
            total += strategy[a] * child_val
        return total

    def _best_response_value(self, state: GameState, reach: List[float]) -> List[float]:
        if state.is_terminal():
            return state.returns()
        player  = state.current_player()
        actions = state.legal_actions()
        info_key = state.info_set_key(player)
        avg_strategy = self.get_average_strategy(info_key, actions)
        n = state.num_players()
        action_vals = {}
        for a in actions:
            new_reach = reach.copy()
            new_reach[player] *= avg_strategy[a]
            action_vals[a] = self._best_response_value(state.apply_action(a), new_reach)
        # Best response: pick action maximizing own value
        best_a = max(actions, key=lambda a: action_vals[a][player])
        return action_vals[best_a]

    def get_policy(self, state: GameState, player: int) -> Dict[int, float]:
        """Get the average (Nash) strategy at a given state."""
        if state.is_terminal(): return {}
        if state.current_player() != player: return {}
        actions = state.legal_actions()
        return self.get_average_strategy(state.info_set_key(player), actions)
