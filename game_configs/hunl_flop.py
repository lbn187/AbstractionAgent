"""HUNL Flop configuration."""

from abstraction_agent.config import GameConfig

HUNL_FLOP_DESCRIPTION = """\
Two-player zero-sum betting game on the second street (flop). This is a
complete rules description; assume the reader has never played this game.

Cards and deal:
- The deck has 52 distinct cards: 13 ranks in order 2 < 3 < 4 < 5 < 6 < 7
  < 8 < 9 < T < J < Q < K < A, and 4 suits: spades, hearts, diamonds,
  clubs. Suits do not have intrinsic rank.
- Each player holds exactly 2 private cards that only they can use.
- There are currently 3 public cards on the board (the "flop").
- After this decision round, 2 more public cards will be revealed one at a
  time (the "turn" card, then the "river" card), with a betting round after
  each. At showdown, there will be 5 public cards total.

Showdown objective:
- At showdown, each player forms the strongest possible 5-card combination
  using any 5 cards from their 7 available cards: 2 private + 5 public.
- The player with the stronger 5-card combination wins the pot.

Hand categories from strongest to weakest:
1. Straight flush  2. Four of a kind  3. Full house  4. Flush
5. Straight  6. Three of a kind  7. Two pair  8. One pair  9. High card

Flop-specific strategic issues:
- Two more public cards will be revealed. Hand rankings can change dramatically.
- "Draws" are holdings that need one or two specific cards to complete a strong
  hand (e.g., four cards to a flush, four consecutive ranks toward a straight).
- Current made hands (pairs, sets, straights, flushes) may be overtaken.
- The flop is the street with the MOST uncertainty about final hand strength.
- Strategic features should capture: current made strength, draw potential
  (flush draws, straight draws, combo draws), vulnerability to future cards,
  board texture interaction, and position relative to the nut hands.
"""

HUNL_FLOP_CONTEXT = """\
Flop endgame. 3 public cards revealed, 2 more to come (turn + river). \
Each player holds 2 private cards. Best 5 of 7 at showdown. \
Hand rankings can change significantly with 2 more cards. \
Draws, made hands, and vulnerability all matter."""

HUNL_FLOP_CONFIG = GameConfig(
    name="hunl_flop",
    game_description=HUNL_FLOP_DESCRIPTION,
    game_context=HUNL_FLOP_CONTEXT,
    n_hands_hint=1176,
    has_future_cards=True,
    future_card_note=(
        "Two more public cards will be revealed (turn then river). "
        "This creates significant uncertainty. Flush draws complete ~35% "
        "of the time, open-ended straight draws ~32%. Many holdings that "
        "are currently weak can become very strong."
    ),
)
