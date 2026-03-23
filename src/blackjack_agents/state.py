"""Pydantic record models and GameStateTracker for recording blackjack games."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .agents.base import CardView

# Hi-Lo card counting values
_HILO: dict[str, int] = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


class ActionRecord(BaseModel):
    """A single decision point during play."""
    player_name: str
    hand_index: int
    hand_total_before: int
    dealer_upcard: CardView
    available_actions: list[str]
    chosen_action: str
    card_received: CardView | None = None
    hand_total_after: int


class HandRecord(BaseModel):
    """Snapshot of a single hand at end of round."""
    cards: list[CardView]
    total: int
    bust: bool
    result: str | None = None
    bet: int
    is_split: bool
    surrendered: bool


class PlayerRoundRecord(BaseModel):
    """One player's data for one round."""
    player_name: str
    agent_type: str
    hands: list[HandRecord]
    actions_taken: list[ActionRecord]


class DealerRoundRecord(BaseModel):
    """Dealer's data for one round."""
    upcard: CardView
    final_hand: list[CardView]
    final_total: int
    bust: bool


class RoundRecord(BaseModel):
    """Complete record of a single round."""
    round_number: int
    players: list[PlayerRoundRecord]
    dealer: DealerRoundRecord
    cards_dealt_count: int
    shoe_remaining: int


class PlayerSummary(BaseModel):
    """Per-player aggregate statistics."""
    player_name: str
    agent_type: str
    total_hands: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blackjacks: int = 0
    busts: int = 0
    surrenders: int = 0
    win_rate: float = 0.0
    net_units: float = 0.0


class ExperimentSummary(BaseModel):
    """Aggregate statistics for an experiment."""
    total_rounds: int
    player_summaries: list[PlayerSummary]


class ExperimentRecord(BaseModel):
    """Complete record of an entire experiment run."""
    experiment_id: str
    timestamp: str
    config: dict[str, Any]
    rounds: list[RoundRecord]
    summary: ExperimentSummary


def _card_view(card: Any) -> CardView:
    """Convert a blackjack21 Card to a CardView."""
    return CardView(suit=str(card.suit), rank=str(card.rank), value=card.value)


class GameStateTracker:
    """Tracks game state across rounds, accumulating records."""

    def __init__(self) -> None:
        self._rounds: list[RoundRecord] = []
        self._current_actions: list[ActionRecord] = []
        self._face_up_cards: list[CardView] = []
        self._round_number: int = 0

    def begin_round(self, round_number: int) -> None:
        self._round_number = round_number
        self._current_actions = []

    def record_action(self, action: ActionRecord) -> None:
        self._current_actions.append(action)

    def record_face_up_card(self, card: CardView) -> None:
        self._face_up_cards.append(card)

    def finalize_round(
        self,
        table: Any,
        agent_type_map: dict[str, str],
        shoe_remaining: int,
    ) -> RoundRecord:
        """Build a RoundRecord from the current table state (must be ROUND_OVER)."""
        from blackjack21 import GameResult

        # Build dealer record
        dealer = table.dealer
        dealer_hand_cards = [_card_view(c) for c in dealer.hand]
        dealer_record = DealerRoundRecord(
            upcard=dealer_hand_cards[0] if dealer_hand_cards else _card_view(dealer.hand[0]),
            final_hand=dealer_hand_cards,
            final_total=dealer.total,
            bust=dealer.bust,
        )

        # Build player records
        total_cards_dealt = len(dealer.hand)
        player_records: list[PlayerRoundRecord] = []
        for player in table.players:
            player_actions = [
                a for a in self._current_actions if a.player_name == player.name
            ]
            hand_records: list[HandRecord] = []
            for hand in player.hands:
                cards = [_card_view(c) for c in hand]
                total_cards_dealt += len(hand)
                result_val = hand.result.value if isinstance(hand.result, GameResult) else None
                hand_records.append(HandRecord(
                    cards=cards,
                    total=hand.total,
                    bust=hand.bust,
                    result=result_val,
                    bet=hand.bet,
                    is_split=len(player.hands) > 1,
                    surrendered=hand.surrendered,
                ))
            player_records.append(PlayerRoundRecord(
                player_name=str(player.name),
                agent_type=agent_type_map.get(str(player.name), "unknown"),
                hands=hand_records,
                actions_taken=player_actions,
            ))

        record = RoundRecord(
            round_number=self._round_number,
            players=player_records,
            dealer=dealer_record,
            cards_dealt_count=total_cards_dealt,
            shoe_remaining=shoe_remaining,
        )
        self._rounds.append(record)
        return record

    def reset_counts(self) -> None:
        """Reset card counting state (called on reshuffle)."""
        self._face_up_cards = []

    @property
    def face_up_cards(self) -> list[CardView]:
        return list(self._face_up_cards)

    @property
    def rounds(self) -> list[RoundRecord]:
        return list(self._rounds)

    @property
    def running_count(self) -> int:
        """Hi-Lo running count based on all face-up cards seen."""
        return sum(_HILO.get(c.rank, 0) for c in self._face_up_cards)

    def true_count(self, remaining_decks: float) -> float:
        """Hi-Lo true count."""
        if remaining_decks <= 0:
            return 0.0
        return self.running_count / remaining_decks
