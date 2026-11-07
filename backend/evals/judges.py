"""Optional LLM-as-judge evaluators (requires OPENAI_API_KEY, --judge flag).

Each judge returns a 1-5 integer score plus a short justification,
following an explicit rubric. Judges are supplementary: the harness's
pass/fail verdicts come from the deterministic assertions, and judge
scores are reported alongside them.
"""

import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RELEVANCE_RUBRIC = """\
Score how well the ANSWER addresses the QUESTION on a 1-5 scale:
5 = fully answers the question, nothing missing, nothing off-topic.
4 = answers the question with minor omissions or minor irrelevant extras.
3 = partially answers; user would need a follow-up.
2 = mostly misses the point but is on topic.
1 = irrelevant or evasive without justification."""

GROUNDEDNESS_RUBRIC = """\
Score how faithful the ANSWER is to the CONTEXT on a 1-5 scale:
5 = every factual claim is directly supported by the context.
4 = all claims supported; small amounts of generic phrasing beyond context.
3 = mostly supported, one unsupported factual claim.
2 = several unsupported claims.
1 = contradicts the context or is largely fabricated."""

TONE_RUBRIC = """\
Score the ANSWER's customer-support tone on a 1-5 scale:
5 = friendly, professional, concise, no jargon leaks (no JSON/system text).
4 = professional with minor stiffness or verbosity.
3 = acceptable but robotic, or slightly too casual.
2 = curt, confusing, or leaks internal artifacts.
1 = rude, nonsensical, or clearly machine-internal output."""


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class LLMJudge:
    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0).with_structured_output(
            JudgeScore
        )

    def _score(self, rubric: str, payload: dict) -> dict:
        prompt = (
            "You are a strict evaluator for a retail support assistant.\n\n"
            f"{rubric}\n\nEvaluate the following and respond with score and justification.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)[:8000]}"
        )
        try:
            result: JudgeScore = self._llm.invoke(prompt)
            return result.model_dump()
        except Exception as exc:  # judge failures must not break the run
            logger.warning("Judge call failed: %s", exc)
            return {"score": None, "justification": f"judge error: {exc}"}

    def relevance(self, question: str, answer: str) -> dict:
        return self._score(RELEVANCE_RUBRIC, {"QUESTION": question, "ANSWER": answer})

    def groundedness(self, answer: str, context: list[str]) -> dict:
        return self._score(GROUNDEDNESS_RUBRIC, {"ANSWER": answer, "CONTEXT": context})

    def tone(self, answer: str) -> dict:
        return self._score(TONE_RUBRIC, {"ANSWER": answer})
