"""Versioned prompt registry."""

from app.agent.prompts.base import PromptPack
from app.agent.prompts.v1 import V1
from app.agent.prompts.v2 import V2

PROMPT_VERSIONS: dict[str, PromptPack] = {p.version: p for p in (V1, V2)}


def get_prompt_pack(version: str) -> PromptPack:
    try:
        return PROMPT_VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"Unknown prompt version '{version}'. Available: {sorted(PROMPT_VERSIONS)}"
        ) from None
