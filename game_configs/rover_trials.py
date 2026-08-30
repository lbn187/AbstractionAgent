"""ROVER TRIALS configuration for the abstraction agent.

ROVER TRIALS is an ORIGINAL game invented for this paper (see rover_trials.py).
Because it appears in no pretraining corpus, the LLM cannot retrieve memorized
strategy knowledge -- it must construct strategic features from the rules text
below. The description states the rules only; it gives NO feature hints (the
agent discovers features zero-shot).

The affinity bonus is a parameter: a LARGE bonus makes affinity a polarized
"draw-like" payoff (huge on the specialized terrain, nothing elsewhere), so that
expected strength (EHS) -- which averages over terrains -- can no longer tell a
volatile draw apart from a steady rover of the same average strength. This is the
regime in which a multi-feature abstraction can beat an EHS baseline.
"""

from abstraction_agent.config import GameConfig


def _description(bonus: int) -> str:
    return f"""\
ROVER TRIALS
A two-player zero-sum betting game of hidden information. This is an ORIGINAL
game; it does not correspond to poker or any standard game.

Setup:
- Each player is privately and independently assigned one ROVER. A rover is
  described by three integer dials, each from 0 to 4:
    * POWER    (0..4): the rover's raw engine strength.
    * GRIP     (0..4): the rover's traction on rough ground.
    * AFFINITY (0..4): the single terrain type the rover is specialized for,
      encoded as: 0=sand, 1=grass, 2=rock, 3=mud, 4=ice.
  There are 5x5x5 = 125 possible rovers. A player sees only their own rover.
- Players do NOT share a deck; both rovers are drawn independently and uniformly,
  so the two players may even hold the same rover.

Play:
- Both players ante 1 chip.
- Betting round 1 happens BEFORE the terrain is known: the first player may
  check or raise (+2 chips); facing a raise a player may fold, call, or
  re-raise; at most 2 raises per round.
- Then the RACE TERRAIN is revealed publicly. It is one of the five terrain
  types, each equally likely (probability 1/5). Terrains sand and grass are
  SMOOTH; rock, mud and ice are ROUGH.
- Betting round 2 happens AFTER the terrain is revealed (raise size +4 chips),
  same betting rules.
- If neither player folds, both rovers race and the one that travels FARTHER
  wins the pot; equal distance splits the pot.

Distance a rover travels on the revealed terrain T:
    distance = POWER
             + GRIP    (added only if T is a ROUGH terrain; contributes 0 on
                        a smooth terrain)
             + {bonus}  (added only if AFFINITY == T, i.e. the terrain revealed
                        is exactly the rover's specialized terrain)
So POWER always contributes; GRIP contributes only on rough terrain; the large
bonus of {bonus} applies only when the revealed terrain matches the rover's
affinity -- so a rover is a "specialist" that is dominant on its own terrain (a
1-in-5 event) but ordinary otherwise.

There are no other rules. Before the terrain is revealed a rover's eventual
distance is uncertain, because it depends on which terrain appears.
"""


def _context(bonus: int) -> str:
    return (
        "ROVER TRIALS (original game): 2-player betting. Your private rover = "
        "(POWER 0-4, GRIP 0-4, AFFINITY 0-4 where 0=sand 1=grass 2=rock 3=mud "
        "4=ice). After betting round 1 a terrain is revealed (5 types, each prob "
        "1/5; sand & grass are smooth, rock/mud/ice are rough). Distance = POWER "
        f"+ (GRIP if terrain is rough) + ({bonus} if AFFINITY == terrain). Farther "
        "rover wins the pot. Two betting rounds, one before and one after the "
        "terrain reveal."
    )


def _future_note(bonus: int) -> str:
    return (
        "After the first betting round, one of five terrains (each equally likely) "
        f"is revealed. GRIP only helps on rough terrain (rock/mud/ice) and the "
        f"large +{bonus} affinity bonus triggers only when the terrain equals the "
        "rover's AFFINITY, so a rover's distance can swing dramatically depending "
        "on which terrain appears (a volatile, draw-like payoff)."
    )


def build_rover_config(affinity_bonus: int = 3) -> GameConfig:
    """GameConfig whose rules text states the given affinity bonus."""
    return GameConfig(
        name=f"rover_trials_b{affinity_bonus}",
        game_description=_description(affinity_bonus),
        game_context=_context(affinity_bonus),
        n_hands_hint=125,
        has_future_cards=True,
        future_card_note=_future_note(affinity_bonus),
    )


# Backward-compatible default (affinity bonus 3, the original ROVER).
ROVER_CONFIG = build_rover_config(3)
