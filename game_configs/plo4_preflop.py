"""PLO4 Preflop configuration."""

from abstraction_agent.config import GameConfig

PLO4_PREFLOP_DESCRIPTION = """\
Multi-player betting game at the preflop stage (before any community cards).
This is a complete rules description; assume the reader has never played.

Cards and deal:
- The deck has 52 distinct cards: 13 ranks in order 2 < 3 < 4 < 5 < 6 < 7
  < 8 < 9 < T < J < Q < K < A, and 4 suits: spades, hearts, diamonds,
  clubs. Suits do not have intrinsic rank.
- Each player holds exactly 4 private cards that only they can see.
- No community cards have been dealt yet.

Post-deal progression:
- After preflop decisions, 5 community cards will be revealed in stages:
  flop (3 cards), turn (1 card), river (1 card), with betting rounds between.
- At showdown, each player forms the strongest possible 5-card combination
  using EXACTLY 2 of their 4 private cards combined with EXACTLY 3 of the
  5 community cards. This is the critical rule: you cannot use 1 or 3 or 4
  private cards; it must be exactly 2 private + 3 community.

Hand categories from strongest to weakest:
1. Straight flush: five consecutive ranks all in the same suit.
2. Four of a kind: four cards of the same rank plus one side card.
3. Full house: three of one rank plus two of another.
4. Flush: five cards of the same suit, not consecutive.
5. Straight: five consecutive ranks, not all same suit.
6. Three of a kind: three of one rank plus two side cards.
7. Two pair: two different pairs plus one side card.
8. One pair: one pair plus three side cards.
9. High card: none of the above.

Preflop strategic considerations (no community cards yet):
- Since no community cards exist, a holding's value is entirely about its
  POTENTIAL to make strong hands after the 5 community cards are revealed.
- Key strategic dimensions for 4-card holdings:
  * SUITEDNESS: How many suits are represented among the 4 cards.
    - Double-suited (2+2): two pairs of suited cards, can make two different
      flushes. Strongest suit structure.
    - Single-suited (3+1 or with one dominant suit pair): one flush draw possible.
    - Rainbow (1+1+1+1): no flush potential from private cards alone.
  * CONNECTIVITY: How close the ranks are to forming straights.
    - Rundown/wrap: 4 consecutive or near-consecutive ranks (e.g., T-9-8-7)
      can make many different straights.
    - Gaps reduce straight combinations; larger gaps are worse.
  * HIGH CARDS AND PAIRS:
    - High pairs (AA, KK) provide immediate top-pair potential.
    - AA with suited/connected side cards is the strongest preflop category.
    - Bare high pairs without coordination are weaker than in 2-card games.
  * NUT POTENTIAL: Whether the holding can make the best possible hand.
    - Nut flush draws (holding the Ace of a suit) are critical.
    - Nut straight potential from connected holdings.
  * CARD COORDINATION: How well all 4 cards work together.
    - All 4 cards contributing to straights/flushes = high coordination.
    - A "dangler" (one card disconnected from the other 3) reduces value.
  * DOMINATION RISK: Whether the holding is likely dominated.
    - Small pairs with no backup are easily dominated.
    - Hands where flush/straight draws can only make non-nut hands.

Approximate hand space:
- There are ~270,725 raw 4-card combinations from 52 cards.
- After suit-isomorphism reduction, ~16,432 strategically distinct holdings.
"""

PLO4_PREFLOP_CONTEXT = """\
Preflop stage of a 4-private-card game. No community cards yet. \
Five community cards will come (3+1+1). At showdown, MUST use exactly \
2 private cards + 3 community cards. Hand value is entirely about \
post-flop potential: suitedness, connectivity, high cards, nut draws, \
and card coordination."""

PLO4_PREFLOP_CONFIG = GameConfig(
    name="plo4_preflop",
    game_description=PLO4_PREFLOP_DESCRIPTION,
    game_context=PLO4_PREFLOP_CONTEXT,
    n_hands_hint=16432,
    has_future_cards=True,
    future_card_note=(
        "Five community cards will be revealed in stages (3+1+1). "
        "At showdown, each player MUST use exactly 2 of their 4 private cards "
        "combined with exactly 3 of the 5 community cards. "
        "Current holding value is entirely about future potential."
    ),
)
