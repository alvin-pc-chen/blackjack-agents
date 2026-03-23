"""GameManager — orchestrates blackjack rounds, dispatches to agents, records state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blackjack21 import Action, GameState, Table

from .agents.base import CardView, GameContext, HandView, OtherPlayerView
from .state import ActionRecord, GameStateTracker, RoundRecord, _card_view

if TYPE_CHECKING:
    from blackjack21 import CardSource

    from .agents.base import Agent

logger = logging.getLogger(__name__)

_ACTION_DISPATCH = {
    Action.HIT: lambda t: t.hit(),
    Action.STAND: lambda t: t.stand(),
    Action.DOUBLE: lambda t: t.double_down(),
    Action.SPLIT: lambda t: t.split(),
    Action.SURRENDER: lambda t: t.surrender(),
}


class GameManager:
    """Drives blackjack rounds using blackjack21, dispatching decisions to Agents."""

    def __init__(
        self,
        player_agents: list[tuple[str, int, Agent]],
        shoe: CardSource,
        *,
        hit_soft_17: bool = False,
        reshuffle_threshold: int = 52,
    ) -> None:
        self._player_tuples = [(name, bet) for name, bet, _ in player_agents]
        self._agent_map: dict[str, Agent] = {name: agent for name, _, agent in player_agents}
        self._agent_type_map: dict[str, str] = {
            name: agent.agent_type for name, _, agent in player_agents
        }
        self._shoe = shoe
        self._hit_soft_17 = hit_soft_17
        self._tracker = GameStateTracker()
        self._round_number = 0
        self._reshuffle_threshold = reshuffle_threshold

        self._table = Table(
            self._player_tuples,
            self._shoe,
            hit_soft_17=hit_soft_17,
        )

    def _check_reshuffle(self) -> None:
        """Reshuffle the shoe if it's below the threshold (like a cut card)."""
        if len(self._shoe) < self._reshuffle_threshold and hasattr(self._shoe, 'reshuffle'):
            logger.info(
                "Shoe down to %d cards (threshold %d), reshuffling",
                len(self._shoe), self._reshuffle_threshold,
            )
            self._shoe.reshuffle()  # type: ignore[union-attr]
            # Reset card counting state since all cards are back in the shoe
            self._tracker.reset_counts()

    def play_round(self) -> RoundRecord:
        """Play a single round start-to-finish, returning the record."""
        self._check_reshuffle()

        self._round_number += 1
        self._tracker.begin_round(self._round_number)

        # Deal initial cards
        self._table.start_game()

        # Record dealer's upcard as face-up
        dealer_visible = self._table.dealer_visible_hand
        if dealer_visible:
            upcard_view = _card_view(dealer_visible[0])
            self._tracker.record_face_up_card(upcard_view)

        # Record all players' initial cards as face-up
        for player in self._table.players:
            for hand in player.hands:
                for card in hand:
                    self._tracker.record_face_up_card(_card_view(card))

        # Drive player turns
        self._drive_player_turns()

        # After all player turns, blackjack21 auto-plays dealer and sets ROUND_OVER.
        # Record dealer's full hand as face-up (the hole card is now revealed).
        if self._table.state == GameState.ROUND_OVER:
            for card in self._table.dealer.hand:
                cv = _card_view(card)
                if cv not in [self._tracker.face_up_cards[-i - 1]
                              for i in range(min(1, len(self._tracker.face_up_cards)))]:
                    pass  # We'll just record all dealer cards below
            # Record hole card (second card, not yet tracked)
            if len(self._table.dealer.hand) >= 2:
                for card in self._table.dealer.hand[1:]:
                    self._tracker.record_face_up_card(_card_view(card))

        return self._tracker.finalize_round(
            self._table,
            self._agent_type_map,
            shoe_remaining=len(self._shoe),
        )

    def play_rounds(self, n: int) -> list[RoundRecord]:
        """Play n rounds sequentially, reshuffling when the shoe runs low."""
        records: list[RoundRecord] = []
        for _ in range(n):
            records.append(self.play_round())
        return records

    def _drive_player_turns(self) -> None:
        """Loop over current player/hand, dispatching to agents."""
        while self._table.state == GameState.PLAYERS_TURN:
            current_player = self._table.current_player
            current_hand = self._table.current_hand
            if current_player is None or current_hand is None:
                break

            available = self._table.available_actions()
            if not available:
                break

            agent = self._agent_map.get(str(current_player.name))
            if agent is None:
                logger.warning("No agent for player %s, standing", current_player.name)
                self._table.stand()
                continue

            context = self._build_context(current_player, current_hand, available)
            action = agent.decide(context)

            # Validate action
            if action not in available:
                logger.warning(
                    "Agent %s returned invalid action %s (available: %s), falling back to STAND",
                    current_player.name, action, available,
                )
                action = Action.STAND

            # Record pre-action state
            total_before = current_hand.total

            # Execute action
            result = _ACTION_DISPATCH[action](self._table)

            # The card received (if any) from hit or double_down
            card_received = None
            if action in (Action.HIT, Action.DOUBLE):
                from blackjack21 import Card as BJ21Card
                if isinstance(result, BJ21Card):
                    card_received = _card_view(result)
                    self._tracker.record_face_up_card(card_received)

            # Record the action
            hand_total_after = current_hand.total
            dealer_upcard = _card_view(self._table.dealer_visible_hand[0])

            self._tracker.record_action(ActionRecord(
                player_name=str(current_player.name),
                hand_index=self._get_hand_index(current_player, current_hand),
                hand_total_before=total_before,
                dealer_upcard=dealer_upcard,
                available_actions=[a.value for a in available],
                chosen_action=action.value,
                card_received=card_received,
                hand_total_after=hand_total_after,
            ))

    def _build_context(self, player: object, hand: object, available: frozenset[Action]) -> GameContext:
        """Build a GameContext snapshot for the agent."""
        from blackjack21 import Player as BJ21Player, Hand as BJ21Hand
        from blackjack21.utils import calculate_hand

        assert isinstance(player, BJ21Player)
        assert isinstance(hand, BJ21Hand)

        hand_cards = [_card_view(c) for c in hand]
        hand_calc = calculate_hand(list(hand))
        dealer_upcard = _card_view(self._table.dealer_visible_hand[0])

        # Other players' visible state
        other_players: list[OtherPlayerView] = []
        for p in self._table.players:
            if str(p.name) == str(player.name):
                continue
            hands: list[HandView] = []
            for h in p.hands:
                hands.append(HandView(
                    cards=[_card_view(c) for c in h],
                    total=h.total,
                    bust=h.bust,
                    num_cards=len(h),
                ))
            other_players.append(OtherPlayerView(name=str(p.name), hands=hands))

        hand_index = self._get_hand_index(player, hand)
        num_decks = getattr(self._shoe, 'num_decks', 6)
        remaining_decks = len(self._shoe) / 52.0

        return GameContext(
            player_name=str(player.name),
            hand_cards=hand_cards,
            hand_total=hand_calc.value,
            hand_is_soft=hand_calc.is_soft,
            hand_index=hand_index,
            num_hands=len(player.hands),
            dealer_upcard=dealer_upcard,
            available_actions=available,
            other_players=other_players,
            round_number=self._round_number,
            face_up_cards=self._tracker.face_up_cards,
            running_count=self._tracker.running_count,
            true_count=self._tracker.true_count(remaining_decks),
            shoe_remaining=len(self._shoe),
        )

    @staticmethod
    def _get_hand_index(player: object, hand: object) -> int:
        from blackjack21 import Player as BJ21Player
        assert isinstance(player, BJ21Player)
        for i, h in enumerate(player.hands):
            if h is hand:
                return i
        return 0

    @property
    def tracker(self) -> GameStateTracker:
        return self._tracker

    @property
    def round_number(self) -> int:
        return self._round_number
