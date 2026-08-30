"""HUNL Turn Endgame configuration."""

from abstraction_agent.config import GameConfig

HUNL_TURN_DESCRIPTION = """\
Two-player zero-sum betting game on the penultimate street. This is a
complete rules description; assume the reader has never played this game.

Cards and deal:
- The deck has 52 distinct cards: 13 ranks in order 2 < 3 < 4 < 5 < 6 < 7
  < 8 < 9 < T < J < Q < K < A, and 4 suits: spades, hearts, diamonds,
  clubs. Suits do not have intrinsic rank.
- Each player holds exactly 2 private cards that only they can use.
- There are currently 4 public cards on the board that both players can use.
- After this decision round, exactly 1 more public card will be drawn
  uniformly from the remaining unseen deck. Then both players will have 2
  private cards plus 5 public cards, 7 total available cards.

Showdown objective:
- At showdown, each player forms the strongest possible 5-card combination
  using any 5 cards from their 7 available cards.
- The player with the stronger 5-card combination wins the pot. If the best
  combinations are exactly tied, the pot is split.

Hand categories from strongest to weakest:
1. Straight flush: five consecutive ranks all in the same suit. The highest
   card of the sequence breaks ties. A-2-3-4-5 is the lowest straight; T-J-Q-K-A
   is the highest straight.
2. Four of a kind: four cards of the same rank plus one side card. Higher
   four-of-a-kind rank wins; if tied, higher side card wins.
3. Full house: three cards of one rank plus two cards of another rank. Higher
   three-of-a-kind rank wins; if tied, higher pair rank wins.
4. Flush: five cards of the same suit, not consecutive. Compare the five ranks
   from highest to lowest until one hand is higher.
5. Straight: five consecutive ranks, not all the same suit. The highest card
   of the sequence breaks ties; A-2-3-4-5 is the lowest straight.
6. Three of a kind: three cards of one rank plus two side cards. Higher trip
   rank wins; if tied, compare side cards from highest to lowest.
7. Two pair: two cards of one rank, two cards of another rank, plus one side
   card. Compare higher pair, then lower pair, then side card.
8. One pair: two cards of one rank plus three side cards. Compare pair rank,
   then side cards from highest to lowest.
9. High card: none of the above. Compare the five ranks from highest to lowest.

Turn-specific strategic issue:
- Because the fifth public card has not been revealed, a holding's value is
  not only its current made strength on the 4-card board.
- Some holdings can improve when the final public card arrives, such as four
  cards toward a flush, four consecutive ranks toward a straight, pairs that
  can become trips/full houses, or made hands that can improve to stronger
  categories.
- Some currently strong holdings are vulnerable: the final public card can
  complete an opponent's straight/flush/full-house possibilities or create
  board patterns that reduce the holding's relative strength.
- Useful strategic features should therefore capture current strength,
  improvement potential, vulnerability, stability across possible final cards,
  and blocker effects from private cards removing possible opponent holdings.
"""

HUNL_TURN_CONTEXT = """\
Penultimate-street endgame. 4 public cards revealed, 1 more random \
card to come. Each player holds 2 private cards. Best 5 of 7 cards \
at showdown. Hand rankings CAN CHANGE when the final card is revealed. \
Current strength, improvement potential, and vulnerability all matter."""

HUNL_TURN_CONFIG = GameConfig(
    name="hunl_turn",
    game_description=HUNL_TURN_DESCRIPTION,
    game_context=HUNL_TURN_CONTEXT,
    n_hands_hint=1000,
    has_future_cards=True,
    future_card_note=(
        "One more public card will be randomly revealed after this round. "
        "This can dramatically change the relative strength of holdings."
    ),
)
