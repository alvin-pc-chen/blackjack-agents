"""Tests for basic strategy agent against known scenarios."""

from blackjack21 import Action

from blackjack_agents.agents.base import CardView
from blackjack_agents.agents.basic_strategy import BasicStrategyAgent
from tests.conftest import make_context


class TestBasicStrategy:
    def setup_method(self) -> None:
        self.agent = BasicStrategyAgent()

    def _dealer(self, rank: str, value: int | None = None) -> CardView:
        vals = {"A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
        return CardView("Diamonds", rank, value or vals[rank])

    def test_hard_16_vs_10_hit(self) -> None:
        ctx = make_context(
            hand_total=16, hand_is_soft=False,
            dealer_upcard=self._dealer("10"),
            available_actions=frozenset({Action.HIT, Action.STAND}),
        )
        # Without surrender available, basic strategy says hit 16 vs 10
        assert self.agent.decide(ctx) == Action.HIT

    def test_hard_16_vs_10_surrender(self) -> None:
        ctx = make_context(
            hand_total=16, hand_is_soft=False,
            dealer_upcard=self._dealer("10"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SURRENDER}),
        )
        assert self.agent.decide(ctx) == Action.SURRENDER

    def test_hard_17_always_stand(self) -> None:
        for dealer_rank in ["2", "6", "7", "10", "A"]:
            ctx = make_context(
                hand_total=17, hand_is_soft=False,
                dealer_upcard=self._dealer(dealer_rank),
            )
            assert self.agent.decide(ctx) == Action.STAND

    def test_hard_11_double(self) -> None:
        ctx = make_context(
            hand_total=11, hand_is_soft=False,
            dealer_upcard=self._dealer("6"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.DOUBLE}),
        )
        assert self.agent.decide(ctx) == Action.DOUBLE

    def test_hard_11_no_double_hits(self) -> None:
        ctx = make_context(
            hand_total=11, hand_is_soft=False,
            dealer_upcard=self._dealer("6"),
            available_actions=frozenset({Action.HIT, Action.STAND}),
        )
        assert self.agent.decide(ctx) == Action.HIT

    def test_hard_12_vs_3_hit(self) -> None:
        ctx = make_context(
            hand_total=12, hand_is_soft=False,
            dealer_upcard=self._dealer("3"),
        )
        assert self.agent.decide(ctx) == Action.HIT

    def test_hard_12_vs_4_stand(self) -> None:
        ctx = make_context(
            hand_total=12, hand_is_soft=False,
            dealer_upcard=self._dealer("4"),
        )
        assert self.agent.decide(ctx) == Action.STAND

    def test_soft_18_vs_6_double(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "A", 11), CardView("Spades", "7", 7)],
            hand_total=18, hand_is_soft=True,
            dealer_upcard=self._dealer("6"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.DOUBLE}),
        )
        assert self.agent.decide(ctx) == Action.DOUBLE

    def test_soft_18_vs_9_hit(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "A", 11), CardView("Spades", "7", 7)],
            hand_total=18, hand_is_soft=True,
            dealer_upcard=self._dealer("9"),
        )
        assert self.agent.decide(ctx) == Action.HIT

    def test_soft_18_vs_7_stand(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "A", 11), CardView("Spades", "7", 7)],
            hand_total=18, hand_is_soft=True,
            dealer_upcard=self._dealer("7"),
        )
        assert self.agent.decide(ctx) == Action.STAND

    def test_pair_8s_split(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "8", 8), CardView("Spades", "8", 8)],
            hand_total=16, hand_is_soft=False,
            dealer_upcard=self._dealer("10"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SPLIT}),
        )
        assert self.agent.decide(ctx) == Action.SPLIT

    def test_pair_10s_no_split(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "K", 10), CardView("Spades", "Q", 10)],
            hand_total=20, hand_is_soft=False,
            dealer_upcard=self._dealer("6"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SPLIT}),
        )
        assert self.agent.decide(ctx) == Action.STAND

    def test_pair_aces_split(self) -> None:
        ctx = make_context(
            hand_cards=[CardView("Hearts", "A", 11), CardView("Spades", "A", 11)],
            hand_total=12,  # soft 12 in blackjack21 (two aces = 11 + 1 = 12)
            hand_is_soft=True,
            dealer_upcard=self._dealer("6"),
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SPLIT}),
        )
        assert self.agent.decide(ctx) == Action.SPLIT

    def test_hard_8_always_hit(self) -> None:
        for dealer_rank in ["2", "6", "10", "A"]:
            ctx = make_context(
                hand_total=8, hand_is_soft=False,
                dealer_upcard=self._dealer(dealer_rank),
            )
            assert self.agent.decide(ctx) == Action.HIT
