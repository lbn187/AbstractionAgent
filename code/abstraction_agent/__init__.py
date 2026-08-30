"""Game-agnostic LLM abstraction agent."""

from .agent import AbstractionAgent
from .config import AgentResult, FeatureDef, GameConfig

__all__ = [
    "AbstractionAgent",
    "AgentResult",
    "FeatureDef",
    "GameConfig",
]
