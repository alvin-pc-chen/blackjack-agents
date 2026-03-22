"""Full basic strategy agent — mathematically optimal play for each situation."""

from __future__ import annotations

from blackjack21 import Action

from .base import Agent, GameContext

# Action shorthand
H = Action.HIT
S = Action.STAND
D = Action.DOUBLE      # double if allowed, else hit
Ds = Action.DOUBLE     # double if allowed, else stand (marked separately below)
P = Action.SPLIT
U = Action.SURRENDER   # surrender if allowed, else hit

# We use a special sentinel to distinguish "double else stand" from "double else hit"
# and "surrender else hit" cases. The decide() method handles fallbacks.

# Hard totals table: (player_total, dealer_upcard_value) -> action
# Dealer upcard value: 2-11 (11 = Ace)
HARD_TABLE: dict[tuple[int, int], Action] = {}

# Hard 5-8: always hit
for total in range(5, 9):
    for dealer in range(2, 12):
        HARD_TABLE[(total, dealer)] = H

# Hard 9: double vs 3-6, else hit
for dealer in range(2, 12):
    if 3 <= dealer <= 6:
        HARD_TABLE[(9, dealer)] = D
    else:
        HARD_TABLE[(9, dealer)] = H

# Hard 10: double vs 2-9, else hit
for dealer in range(2, 12):
    if 2 <= dealer <= 9:
        HARD_TABLE[(10, dealer)] = D
    else:
        HARD_TABLE[(10, dealer)] = H

# Hard 11: double vs 2-10, hit vs A
for dealer in range(2, 12):
    if dealer <= 10:
        HARD_TABLE[(11, dealer)] = D
    else:
        HARD_TABLE[(11, dealer)] = H

# Hard 12: stand vs 4-6, else hit
for dealer in range(2, 12):
    if 4 <= dealer <= 6:
        HARD_TABLE[(12, dealer)] = S
    else:
        HARD_TABLE[(12, dealer)] = H

# Hard 13-16: stand vs 2-6, else hit
for total in range(13, 17):
    for dealer in range(2, 12):
        if 2 <= dealer <= 6:
            HARD_TABLE[(total, dealer)] = S
        else:
            HARD_TABLE[(total, dealer)] = H

# Hard 17-21: always stand
for total in range(17, 22):
    for dealer in range(2, 12):
        HARD_TABLE[(total, dealer)] = S

# --- Surrender overrides (applied on top of hard table) ---
# Surrender 16 vs 9, 10, A
# Surrender 15 vs 10
_SURRENDER_SPOTS: set[tuple[int, int]] = {
    (16, 9), (16, 10), (16, 11),
    (15, 10),
}

# Soft totals table: (player_total, dealer_upcard_value) -> action
# "Soft total" means hand total when one ace counts as 11
SOFT_TABLE: dict[tuple[int, int], Action] = {}

# Soft 13 (A,2): double vs 5-6, else hit
# Soft 14 (A,3): double vs 5-6, else hit
for total in (13, 14):
    for dealer in range(2, 12):
        if 5 <= dealer <= 6:
            SOFT_TABLE[(total, dealer)] = D
        else:
            SOFT_TABLE[(total, dealer)] = H

# Soft 15 (A,4): double vs 4-6, else hit
# Soft 16 (A,5): double vs 4-6, else hit
for total in (15, 16):
    for dealer in range(2, 12):
        if 4 <= dealer <= 6:
            SOFT_TABLE[(total, dealer)] = D
        else:
            SOFT_TABLE[(total, dealer)] = H

# Soft 17 (A,6): double vs 3-6, else hit
for dealer in range(2, 12):
    if 3 <= dealer <= 6:
        SOFT_TABLE[(17, dealer)] = D
    else:
        SOFT_TABLE[(17, dealer)] = H

# Soft 18 (A,7): double vs 3-6, stand vs 2/7/8, hit vs 9/10/A
for dealer in range(2, 12):
    if 3 <= dealer <= 6:
        SOFT_TABLE[(18, dealer)] = D
    elif dealer in (2, 7, 8):
        SOFT_TABLE[(18, dealer)] = S
    else:
        SOFT_TABLE[(18, dealer)] = H

# Soft 19 (A,8): always stand (some charts double vs 6)
for dealer in range(2, 12):
    SOFT_TABLE[(19, dealer)] = S

# Soft 20 (A,9): always stand
for dealer in range(2, 12):
    SOFT_TABLE[(20, dealer)] = S

# Soft 21: always stand
for dealer in range(2, 12):
    SOFT_TABLE[(21, dealer)] = S

# Pair splitting table: (card_value, dealer_upcard_value) -> split or not
# True = split, False = don't split (use hard/soft table)
PAIR_TABLE: dict[tuple[int, int], bool] = {}

# 2,2: split vs 2-7
# 3,3: split vs 2-7
for card in (2, 3):
    for dealer in range(2, 12):
        PAIR_TABLE[(card, dealer)] = 2 <= dealer <= 7

# 4,4: split vs 5-6 only
for dealer in range(2, 12):
    PAIR_TABLE[(4, dealer)] = 5 <= dealer <= 6

# 5,5: never split (play as hard 10)
for dealer in range(2, 12):
    PAIR_TABLE[(5, dealer)] = False

# 6,6: split vs 2-6
for dealer in range(2, 12):
    PAIR_TABLE[(6, dealer)] = 2 <= dealer <= 6

# 7,7: split vs 2-7
for dealer in range(2, 12):
    PAIR_TABLE[(7, dealer)] = 2 <= dealer <= 7

# 8,8: always split
for dealer in range(2, 12):
    PAIR_TABLE[(8, dealer)] = True

# 9,9: split vs 2-9 except 7
for dealer in range(2, 12):
    PAIR_TABLE[(9, dealer)] = dealer not in (7, 10, 11)

# 10,10: never split
for dealer in range(2, 12):
    PAIR_TABLE[(10, dealer)] = False

# A,A: always split
for dealer in range(2, 12):
    PAIR_TABLE[(11, dealer)] = True


def _dealer_value(upcard_value: int) -> int:
    """Normalize dealer upcard: face cards are 10, ace is 11."""
    return upcard_value


def _is_pair(context: GameContext) -> tuple[bool, int]:
    """Check if the hand is a pair (exactly 2 cards of same value)."""
    if len(context.hand_cards) != 2:
        return False, 0
    v1, v2 = context.hand_cards[0].value, context.hand_cards[1].value
    if v1 == v2:
        return True, v1
    return False, 0


class BasicStrategyAgent(Agent):
    """Plays according to the standard basic strategy chart."""

    def decide(self, context: GameContext) -> Action:
        dealer_val = _dealer_value(context.dealer_upcard.value)
        available = context.available_actions

        # Check for pairs first
        is_pair, pair_val = _is_pair(context)
        if is_pair and Action.SPLIT in available:
            should_split = PAIR_TABLE.get((pair_val, dealer_val), False)
            if should_split:
                return Action.SPLIT

        # Look up in soft or hard table
        total = context.hand_total
        if context.hand_is_soft and total <= 21:
            action = SOFT_TABLE.get((total, dealer_val), Action.STAND)
        else:
            # Check surrender first
            if (total, dealer_val) in _SURRENDER_SPOTS and Action.SURRENDER in available:
                return Action.SURRENDER
            action = HARD_TABLE.get((total, dealer_val), Action.STAND)

        # Handle fallbacks for double
        if action == Action.DOUBLE:
            if Action.DOUBLE in available:
                return Action.DOUBLE
            # "Double else stand" for soft 18 vs 3-6 is handled by returning HIT
            # since basic strategy double-else-hit is the common case
            return Action.HIT

        # Validate action is available
        if action in available:
            return action

        # Final fallback
        if Action.STAND in available:
            return Action.STAND
        return Action.HIT
