"""
Game Engine: Leduc Hold'em + Asymmetric Auction Game
Both are extensive-form games with imperfect information.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
#  Abstract Base
# ─────────────────────────────────────────────

class GameState(ABC):
    @abstractmethod
    def is_terminal(self) -> bool: ...
    @abstractmethod
    def current_player(self) -> int: ...
    @abstractmethod
    def legal_actions(self) -> List[int]: ...
    @abstractmethod
    def apply_action(self, action: int) -> "GameState": ...
    @abstractmethod
    def returns(self) -> List[float]: ...
    @abstractmethod
    def info_set_key(self, player: int) -> str: ...
    @abstractmethod
    def info_set_tensor(self, player: int) -> np.ndarray: ...
    @abstractmethod
    def num_players(self) -> int: ...


# ─────────────────────────────────────────────
#  GAME 1: Leduc Hold'em
#  2-player poker with 6 cards (J,Q,K × 2 suits)
#  Two betting rounds; community card revealed in round 2
# ─────────────────────────────────────────────

class LeducAction(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2

LEDUC_RANKS = ['J', 'Q', 'K']
LEDUC_SUITS = [0, 1]
LEDUC_DECK  = [(r, s) for r in range(3) for s in LEDUC_SUITS]  # 6 cards

def leduc_hand_strength(hole: Tuple, community: Optional[Tuple]) -> int:
    """Higher = better. Pair > high card."""
    if community is None:
        return hole[0]  # just rank
    if hole[0] == community[0]:
        return 10 + hole[0]   # pair
    return hole[0]

@dataclass
class LeducState(GameState):
    hole_cards: List[Optional[Tuple]]  = field(default_factory=lambda: [None, None])
    community_card: Optional[Tuple]    = None
    pot: float                          = 2.0   # antes
    bets: List[float]                  = field(default_factory=lambda: [1.0, 1.0])
    street: int                        = 0      # 0=preflop, 1=flop
    acting_player: int                 = 0
    history: List[int]                 = field(default_factory=list)
    _deck: List                        = field(default_factory=lambda: list(range(6)))
    _rng: np.random.Generator          = field(default_factory=np.random.default_rng)
    _folded: Optional[int]             = None
    _street_actions: int               = 0
    _last_bet: bool                    = False

    @staticmethod
    def new_game(seed: Optional[int] = None) -> "LeducState":
        rng   = np.random.default_rng(seed)
        deck  = list(range(6))
        rng.shuffle(deck)
        state = LeducState()
        state._rng   = rng
        state._deck  = deck
        state.hole_cards = [LEDUC_DECK[deck[0]], LEDUC_DECK[deck[1]]]
        state.pot    = 2.0
        state.bets   = [1.0, 1.0]
        return state

    def num_players(self) -> int: return 2

    def is_terminal(self) -> bool:
        return self._folded is not None or (self.street == 2)

    def current_player(self) -> int: return self.acting_player

    def legal_actions(self) -> List[int]:
        if self.is_terminal(): return []
        return [LeducAction.FOLD, LeducAction.CHECK_CALL, LeducAction.BET_RAISE]

    def apply_action(self, action: int) -> "LeducState":
        import copy
        s = copy.deepcopy(self)
        s.history = s.history + [action]

        if action == LeducAction.FOLD:
            s._folded = s.acting_player
            return s

        if action == LeducAction.CHECK_CALL:
            # call if there's a bet to call
            diff = max(s.bets) - s.bets[s.acting_player]
            s.bets[s.acting_player] += diff
            s.pot += diff
            s._last_bet = False
        elif action == LeducAction.BET_RAISE:
            bet_size = 2.0 if s.street == 0 else 4.0
            prev_bet = s.bets[s.acting_player]
            # call the difference first, then raise by bet_size
            s.bets[s.acting_player] = max(s.bets) + bet_size
            # pot increases by only the new chips put in by this player
            s.pot += s.bets[s.acting_player] - prev_bet
            s._last_bet = True

        s._street_actions += 1
        opp = 1 - s.acting_player

        # Advance street: after both players have acted and bets are equal
        street_over = (
            s._street_actions >= 2 and
            (not s._last_bet or s._street_actions >= 4)
        )

        if street_over:
            if s.street == 0:
                # reveal community card
                s.community_card = LEDUC_DECK[s._deck[2]]
                s.street = 1
                s._street_actions = 0
                s._last_bet = False
                s.acting_player = 0
            else:
                s.street = 2  # terminal
        else:
            s.acting_player = opp

        return s

    def returns(self) -> List[float]:
        if self._folded is not None:
            winner = 1 - self._folded
            loser  = self._folded
            r = [0.0, 0.0]
            r[winner] = self.bets[loser]
            r[loser]  = -self.bets[loser]
            return r
        # showdown
        s0 = leduc_hand_strength(self.hole_cards[0], self.community_card)
        s1 = leduc_hand_strength(self.hole_cards[1], self.community_card)
        if s0 > s1:   return [self.bets[1], -self.bets[1]]
        if s1 > s0:   return [-self.bets[0], self.bets[0]]
        return [0.0, 0.0]  # tie

    def info_set_key(self, player: int) -> str:
        hole = self.hole_cards[player]
        comm = self.community_card if self.street >= 1 else None
        h_str = f"{LEDUC_RANKS[hole[0]]}{hole[1]}"
        c_str = f"{LEDUC_RANKS[comm[0]]}{comm[1]}" if comm else "?"
        act_str = "".join(["F","C","R"][a] for a in self.history)
        return f"{h_str}|{c_str}|{act_str}"

    def info_set_tensor(self, player: int) -> np.ndarray:
        """Encode info set as a fixed-length float vector (dim=30)."""
        vec = np.zeros(30, dtype=np.float32)
        # Hole card one-hot (6 cards)
        if self.hole_cards[player] is not None:
            rank, suit = self.hole_cards[player]
            vec[rank * 2 + suit] = 1.0
        # Community card one-hot (6 cards, offset 6)
        if self.community_card is not None:
            rank, suit = self.community_card
            vec[6 + rank * 2 + suit] = 1.0
        # Street (1 bit, offset 12)
        vec[12] = float(self.street)
        # Pot size (normalized, offset 13)
        vec[13] = self.pot / 20.0
        # History encoding (up to 8 actions × 3, offset 14)
        for i, a in enumerate(self.history[-8:]):
            idx = 14 + i * 3 + a
            if idx < len(vec):
                vec[idx] = 1.0
        return vec


# ─────────────────────────────────────────────
#  GAME 2: Asymmetric Auction (novel extension)
#  1 Seller sets price, 2 Buyers with private values
#  Seller wants to maximize revenue; buyers want surplus
#  Different action spaces per player type
# ─────────────────────────────────────────────

class AuctionRole(IntEnum):
    SELLER = 0
    BUYER_A = 1
    BUYER_B = 2

SELLER_PRICES  = [2, 4, 6, 8, 10]   # seller's action space
BUYER_ACTIONS  = [0, 1]              # 0=reject, 1=accept

@dataclass
class AuctionState(GameState):
    private_values: List[int]    = field(default_factory=lambda: [0, 0, 0])
    posted_price: Optional[int]  = None
    decisions: List[Optional[int]] = field(default_factory=lambda: [None, None])
    phase: int                   = 0   # 0=seller prices, 1=buyer A, 2=buyer B, 3=terminal
    history: List[int]           = field(default_factory=list)

    @staticmethod
    def new_game(seed: Optional[int] = None) -> "AuctionState":
        rng = np.random.default_rng(seed)
        # Seller's "value" = production cost (private, low)
        # Buyers' values = willingness to pay (private, varied)
        seller_cost = int(rng.integers(1, 4))
        buyer_a_val = int(rng.integers(3, 11))
        buyer_b_val = int(rng.integers(3, 11))
        s = AuctionState()
        s.private_values = [seller_cost, buyer_a_val, buyer_b_val]
        return s

    def num_players(self) -> int: return 3

    def is_terminal(self) -> bool: return self.phase == 3

    def current_player(self) -> int: return self.phase  # phase maps to player

    def legal_actions(self) -> List[int]:
        if self.phase == 0: return list(range(len(SELLER_PRICES)))
        if self.phase in (1, 2): return BUYER_ACTIONS
        return []

    def apply_action(self, action: int) -> "AuctionState":
        import copy
        s = copy.deepcopy(self)
        s.history = s.history + [action]
        if s.phase == 0:
            s.posted_price = SELLER_PRICES[action]
            s.phase = 1
        elif s.phase == 1:
            s.decisions[0] = action
            s.phase = 2
        elif s.phase == 2:
            s.decisions[1] = action
            s.phase = 3
        return s

    def returns(self) -> List[float]:
        price = self.posted_price
        cost  = self.private_values[0]
        v_a   = self.private_values[1]
        v_b   = self.private_values[2]
        accept_a = (self.decisions[0] == 1 and v_a >= price)
        accept_b = (self.decisions[1] == 1 and v_b >= price)
        num_sold = int(accept_a) + int(accept_b)
        seller_profit = num_sold * (price - cost)
        buyer_a_surplus = (v_a - price) if accept_a else 0
        buyer_b_surplus = (v_b - price) if accept_b else 0
        return [float(seller_profit), float(buyer_a_surplus), float(buyer_b_surplus)]

    def info_set_key(self, player: int) -> str:
        pv = self.private_values[player]
        price_str = str(self.posted_price) if self.posted_price else "?"
        act_str = "".join(str(a) for a in self.history)
        if player == AuctionRole.SELLER:
            return f"S:cost={pv}|hist={act_str}"
        return f"B{player}:val={pv}|price={price_str}|hist={act_str}"

    def info_set_tensor(self, player: int) -> np.ndarray:
        """Encode as float vector (dim=26). Player-conditioned.
        Layout: [player_onehot(3), private_val(1), price(1), phase(1), history(4×5=20)]
        Total: 3+1+1+1+20 = 26
        """
        vec = np.zeros(26, dtype=np.float32)
        # Player one-hot (3, offset 0)
        vec[player] = 1.0
        # Private value normalized (offset 3)
        vec[3] = self.private_values[player] / 10.0
        # Posted price (if known, offset 4)
        if self.posted_price is not None:
            vec[4] = self.posted_price / 10.0
        # Phase (offset 5)
        vec[5] = self.phase / 3.0
        # History (up to 4 actions × 5, offset 6; max idx = 6+3*5+4 = 25, within size 26)
        for i, a in enumerate(self.history[-4:]):
            vec[6 + i * 5 + a] = 1.0
        return vec
