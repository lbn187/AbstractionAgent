"""Card representation and utility functions.

Card encoding (matching src/Card.hpp):
  index = number * 4 + suit
  number: 0=A, 1=K, 2=Q, 3=J, 4=T, 5=9, 6=8, 7=7, 8=6, 9=5, 10=4, 11=3, 12=2
  suit: 0=s, 1=d, 2=c, 3=h
"""

from __future__ import annotations

from itertools import combinations
from random import Random
from typing import List, Tuple

Hand = Tuple[int, ...]

NUMBER_NAMES = "AKQJT98765432"
SUIT_NAMES = "sdch"
CARD_COUNT = 52


def card_to_str(card_id: int) -> str:
    """Convert card index to human-readable string like 'Ah'."""
    return NUMBER_NAMES[card_id // 4] + SUIT_NAMES[card_id % 4]


def str_to_card(s: str) -> int:
    """Convert 'Ah' style string to card index."""
    return NUMBER_NAMES.index(s[0]) * 4 + SUIT_NAMES.index(s[1])


def hand_to_str(hand: Hand) -> str:
    """Convert hand tuple to readable string."""
    return " ".join(card_to_str(c) for c in hand)


def card_number(card_id: int) -> int:
    """Return the number (rank) of the card, 0=A, 12=2."""
    return card_id // 4


def card_suit(card_id: int) -> int:
    """Return the suit of the card."""
    return card_id % 4


def hands_conflict(hand_a: Hand, hand_b: Hand) -> bool:
    """Check if two hands share any card."""
    return bool(set(hand_a) & set(hand_b))


def hand_conflicts_with_board(hand: Hand, board: List[int]) -> bool:
    """Check if a hand shares any card with the board."""
    board_set = set(board)
    return any(c in board_set for c in hand)


def enumerate_hunl_hands(board: List[int]) -> List[Hand]:
    """Enumerate all valid 2-card HUNL hands that don't conflict with board."""
    board_set = set(board)
    hands: List[Hand] = []
    for c1 in range(CARD_COUNT):
        if c1 in board_set:
            continue
        for c2 in range(c1 + 1, CARD_COUNT):
            if c2 in board_set:
                continue
            hands.append((c1, c2))
    return sorted(hands)


def enumerate_plo4_hands(board: List[int]) -> List[Hand]:
    """Enumerate all valid 4-card PLO4 hands that don't conflict with board."""
    remaining = [c for c in range(CARD_COUNT) if c not in board]
    return sorted(combinations(remaining, 4))


def random_board(rng: Random, n_cards: int = 5) -> List[int]:
    """Generate a random board of n_cards cards."""
    return sorted(rng.sample(range(CARD_COUNT), n_cards))
