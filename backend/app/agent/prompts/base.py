"""Prompt pack structure shared by all prompt versions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPack:
    """A complete, versioned set of prompts for the agent.

    Versions are compared experimentally with the evaluation harness
    (``python -m evals.run_eval --prompt-version v1`` vs ``v2``).
    """

    version: str
    description: str
    router_system: str
    direct_system: str
    # str.format template with {context} placeholder for retrieved passages.
    rag_system_template: str
    tool_system: str
