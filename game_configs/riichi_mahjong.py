"""Riichi Mahjong (Japanese Competitive Mahjong) configuration."""

from abstraction_agent.config import GameConfig

RIICHI_MAHJONG_DESCRIPTION = """\
Two-player zero-sum imperfect-information game: Japanese Riichi Mahjong (竞技麻将).
This is a complete rules description for the mid-game decision point.

Tiles:
- 136 tiles total: 34 unique tile types, 4 copies each.
- Number tiles (数牌): three suits of 1-9:
  * Man (万子/Characters): 1m-9m
  * Pin (筒子/Circles): 1p-9p
  * Sou (索子/Bamboo): 1s-9s
- Honor tiles (字牌):
  * Wind tiles (风牌): East(東), South(南), West(西), North(北)
  * Dragon tiles (三元牌): White(白), Green(發), Red(中)

Winning condition:
- A complete hand consists of 14 tiles arranged as:
  4 sets (mentsu) + 1 pair (jantai), where each set is either:
  * Sequence (shuntsu): three consecutive tiles of the same number suit (e.g., 2m3m4m)
  * Triplet (koutsu): three identical tiles (e.g., 5p5p5p)
- Special winning patterns: Seven Pairs (七対子), Thirteen Orphans (国士無双)

Shanten (向聴数):
- The minimum number of tile exchanges needed to reach tenpai (ready to win).
- Tenpai (聴牌): shanten = 0, meaning one more tile completes the hand.
- Lower shanten = closer to winning = strategically stronger.

Key strategic dimensions in mid-game:
1. HAND PROGRESSION (向聴数/進行):
   - Shanten number determines how close to winning.
   - Number of effective tiles (有効牌) that reduce shanten.
   - Multiple paths to tenpai vs. single narrow path.

2. HAND VALUE POTENTIAL (打点):
   - Yaku (役) possibilities: scoring patterns that determine hand value.
   - Common yaku: Riichi, Tanyao (all simples), Pinfu (no-points),
     Iipeiko (pure double sequence), Yakuhai (value tiles).
   - Higher-value yaku: Honitsu (half flush), Chinitsu (full flush),
     Toitoi (all triplets), Sanshoku (mixed triple sequence).
   - Han (翻) count determines payment: 1 han ≈ 1000pts, 3 han ≈ 4000pts,
     mangan (5+ han) = 8000pts, haneman (6-7) = 12000pts.

3. TILE EFFICIENCY (牌効率):
   - Acceptance count (受入枚数): how many tiles in the wall improve the hand.
   - Wider acceptance = faster progression = higher expected value.
   - Trade-off between acceptance width and hand value.

4. DEFENSIVE SAFETY (守備力):
   - Safety of discards: likelihood of dealing into opponent's winning hand.
   - Safe tiles: tiles already discarded by opponent (現物), honor tiles with
     3+ visible copies, tiles adjacent to opponent's discards (壁/スジ).
   - Hand shape flexibility for defensive retreat.

5. WAITING PATTERN (待ち):
   - When tenpai, the specific tiles that complete the hand.
   - Good waits: many remaining copies in wall, hard for opponent to read.
   - Bad waits: few remaining copies, easily readable from discards.

6. FUTURE DRAW POTENTIAL (ツモ期待):
   - Remaining tiles in the wall that improve the hand.
   - Probability of drawing useful tiles given visible information.
   - This is the primary CHANCE NODE: each draw from the wall is random.

Game flow with chance nodes:
- Each turn, a player draws one tile from the wall (CHANCE NODE).
- Then decides: keep the drawn tile and discard another, or discard the drawn tile.
- If the hand becomes complete (tsumo), the player wins.
- If an opponent's discard completes your hand (ron), you can claim it.
- The wall has ~70 drawable tiles at game start; mid-game ~40-50 remain.

Current scenario:
- Mid-game position (turn 8 of East round).
- Each player holds 13 tiles (waiting to draw the 14th).
- Some tiles are visible (discards, dora indicators).
- The strategic question: given your 13-tile hand and visible information,
  how should hands be grouped by strategic similarity?
"""

RIICHI_MAHJONG_CONTEXT = """\
Mid-game Riichi Mahjong (turn 8, East round). Player holds 13 tiles. \
Future draws from the wall are chance nodes that can improve or worsen \
the hand. Strategic value depends on: shanten (distance to win), \
acceptance width (number of useful draws), hand value potential (yaku/han), \
defensive safety, and waiting pattern quality when tenpai."""

RIICHI_MAHJONG_CONFIG = GameConfig(
    name="riichi_mahjong",
    game_description=RIICHI_MAHJONG_DESCRIPTION,
    game_context=RIICHI_MAHJONG_CONTEXT,
    n_hands_hint=200,
    has_future_cards=True,
    future_card_note=(
        "Each turn, one tile is drawn randomly from the remaining wall "
        "(~40-50 tiles remain in mid-game). This draw can dramatically change "
        "hand value: completing a winning hand, reducing shanten, or enabling "
        "higher-value yaku. The draw is the primary chance node."
    ),
)
