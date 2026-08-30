"""HUNL Preflop configuration."""

from abstraction_agent.config import GameConfig

HUNL_PREFLOP_DESCRIPTION = """\
Two-player zero-sum betting game before any public cards are revealed
(preflop). This is a complete rules description; assume the reader has
never played this game.

Cards and deal:
- The deck has 52 distinct cards: 13 ranks in order 2 < 3 < 4 < 5 < 6 < 7
  < 8 < 9 < T < J < Q < K < A, and 4 suits: spades, hearts, diamonds,
  clubs. Suits do not have intrinsic rank.
- Each player holds exactly 2 private cards that only they can use.
- No public cards have been revealed yet.
- After this decision round, 5 public cards will be revealed over three
  streets (3 on the flop, 1 on the turn, 1 on the river), with a betting
  round after each.

Showdown objective:
- At showdown, each player forms the strongest possible 5-card combination
  using any 5 cards from their 7 available cards: 2 private + 5 public.
- The player with the stronger 5-card combination wins the pot.

Hand categories from strongest to weakest:
1. Straight flush  2. Four of a kind  3. Full house  4. Flush
5. Straight  6. Three of a kind  7. Two pair  8. One pair  9. High card

Preflop-specific strategic issues:
- No public cards exist yet. All evaluation is about potential rather than
  current made hands.
- Only 169 strategically distinct starting hand types exist (considering
  that suits are interchangeable preflop):
  * 13 pocket pairs: 22, 33, ..., AA
  * 78 suited non-pairs: A2s, A3s, ..., KQs
  * 78 offsuit non-pairs: A2o, A3o, ..., KQo
- "s" means both cards share a suit (flush potential); "o" means different
  suits.
- Strategic features should capture: raw card strength (high cards), pair
  potential, suited/connected properties, domination relationships, and
  how well the hand realizes equity across different board runouts.
"""

HUNL_PREFLOP_CONTEXT = """\
Preflop. No public cards yet, 5 to come over 3 streets. \
Each player holds 2 private cards. 169 distinct starting hand types. \
All evaluation is about long-run potential. \
Pairs, high cards, suitedness, and connectedness are primary factors."""

HUNL_PREFLOP_CONFIG = GameConfig(
    name="hunl_preflop",
    game_description=HUNL_PREFLOP_DESCRIPTION,
    game_context=HUNL_PREFLOP_CONTEXT,
    n_hands_hint=169,
    has_future_cards=True,
    future_card_note=(
        "Five public cards will be revealed over three streets. "
        "Hand strength is entirely about potential. Suited hands gain "
        "flush equity, connected hands gain straight equity, pairs have "
        "set-mining potential."
    ),
)
