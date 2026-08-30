"""Game configurations used in the paper (natural-language rules as input)."""

from .hunl_flop import HUNL_FLOP_CONFIG
from .hunl_preflop import HUNL_PREFLOP_CONFIG
from .hunl_turn import HUNL_TURN_CONFIG
from .plo4_preflop import PLO4_PREFLOP_CONFIG
from .riichi_mahjong import RIICHI_MAHJONG_CONFIG
from .rover_trials import ROVER_CONFIG, build_rover_config

GAME_CONFIGS = {
    "hunl_flop": HUNL_FLOP_CONFIG,
    "hunl_preflop": HUNL_PREFLOP_CONFIG,
    "hunl_turn": HUNL_TURN_CONFIG,
    "plo4_preflop": PLO4_PREFLOP_CONFIG,
    "riichi_mahjong": RIICHI_MAHJONG_CONFIG,
    "rover_trials": ROVER_CONFIG,
}

__all__ = [
    "GAME_CONFIGS",
    "HUNL_FLOP_CONFIG",
    "HUNL_PREFLOP_CONFIG",
    "HUNL_TURN_CONFIG",
    "PLO4_PREFLOP_CONFIG",
    "RIICHI_MAHJONG_CONFIG",
    "ROVER_CONFIG",
    "build_rover_config",
]
