"""Tests for shoe module."""

from blackjack21 import EmptyDeckError
import pytest

from blackjack_agents.shoe import PredeterminedShoe, SeededShoe, create_shoe
from .conftest import make_card


class TestSeededShoe:
    def test_deterministic_order(self) -> None:
        s1 = SeededShoe(seed=42, num_decks=6)
        s2 = SeededShoe(seed=42, num_decks=6)
        cards1 = [s1.draw_card() for _ in range(50)]
        cards2 = [s2.draw_card() for _ in range(50)]
        assert all(
            c1.rank == c2.rank and c1.suit == c2.suit
            for c1, c2 in zip(cards1, cards2)
        )

    def test_different_seeds_differ(self) -> None:
        s1 = SeededShoe(seed=42, num_decks=6)
        s2 = SeededShoe(seed=99, num_decks=6)
        cards1 = [s1.draw_card() for _ in range(20)]
        cards2 = [s2.draw_card() for _ in range(20)]
        assert any(
            c1.rank != c2.rank or c1.suit != c2.suit
            for c1, c2 in zip(cards1, cards2)
        )

    def test_correct_card_count(self) -> None:
        shoe = SeededShoe(seed=1, num_decks=6)
        assert len(shoe) == 312  # 52 * 6

    def test_draw_reduces_length(self) -> None:
        shoe = SeededShoe(seed=1, num_decks=1)
        assert len(shoe) == 52
        shoe.draw_card()
        assert len(shoe) == 51

    def test_drawn_cards_tracked(self) -> None:
        shoe = SeededShoe(seed=1, num_decks=1)
        drawn = [shoe.draw_card() for _ in range(5)]
        assert len(shoe.drawn_cards) == 5
        assert all(d.rank == s.rank for d, s in zip(drawn, shoe.drawn_cards))

    def test_initial_order_preserved(self) -> None:
        shoe = SeededShoe(seed=1, num_decks=1)
        initial = shoe.initial_order
        for _ in range(10):
            shoe.draw_card()
        assert shoe.initial_order == initial

    def test_empty_shoe_raises(self) -> None:
        shoe = SeededShoe(seed=1, num_decks=1)
        for _ in range(52):
            shoe.draw_card()
        with pytest.raises(EmptyDeckError):
            shoe.draw_card()


class TestPredeterminedShoe:
    def test_deals_in_order(self) -> None:
        cards = [make_card("A"), make_card("K"), make_card("5")]
        shoe = PredeterminedShoe(cards)
        assert shoe.draw_card().rank == "A"
        assert shoe.draw_card().rank == "K"
        assert shoe.draw_card().rank == "5"

    def test_empty_raises(self) -> None:
        shoe = PredeterminedShoe([make_card("A")])
        shoe.draw_card()
        with pytest.raises(EmptyDeckError):
            shoe.draw_card()


class TestCreateShoe:
    def test_with_seed(self) -> None:
        shoe = create_shoe(seed=42, num_decks=2)
        assert isinstance(shoe, SeededShoe)
        assert len(shoe) == 104

    def test_with_cards(self) -> None:
        cards = [make_card("A"), make_card("K")]
        shoe = create_shoe(cards=cards)
        assert isinstance(shoe, PredeterminedShoe)
        assert len(shoe) == 2

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError):
            create_shoe()
