"""
agents/judge_agent.py
---------------------
The Judge Agent independently scores both LLM responses on the shared
8-factor rubric (agents/rubric.py) - the SAME rubric the Verifier uses -
then selects the better one.

Evaluation Criteria (8 dimensions, each scored 0-10 - see agents/rubric.py):
  1. Factual Correctness
  2. Question Relevance
  3. Context Relevance
  4. Faithfulness / Groundedness
  5. Completeness
  6. Hallucination-Free
  7. Clarity
  8. Consistency

Explainability:
  Every score is accompanied by a one-sentence justification quoting or
  referencing the specific part of the answer that drove the score.

The Judge scores fully blind, with no other evaluator's scores shown to it
(see build_judge_prompt below - this was previously a bug, see
RESEARCH_LOG.md Section 12).

Answer identity ("Answer 1" / "Answer 2") is intentionally generic, never
"LLaMA 3" / "Mistral" - which physical model produced which answer is
randomized per question by the caller (generate_training_data_v2.py /
server.py) precisely so the Judge can never learn "Answer 1 = LLaMA 3"
across the dataset. Revealing model identity would confound position bias
with brand-identity bias, and earlier data (before this fix) showed a
strong, judge-model-dependent skew toward whichever position was always
LLaMA 3/Mistral - see RESEARCH_LOG.md Section 17.

Output format (structured for frontend/DB parsing):
  ## Answer 1
  - Factual Correctness: X/10 | <one sentence why>
  - Question Relevance:  X/10 | <one sentence why>
  - Context Relevance:   X/10 | <one sentence why>
  - Faithfulness:        X/10 | <one sentence why>
  - Completeness:        X/10 | <one sentence why>
  - Hallucination-Free:  X/10 | <one sentence why>
  - Clarity:             X/10 | <one sentence why>
  - Consistency:         X/10 | <one sentence why>
  - TOTAL: XX/80

  ## Answer 2
  (same 8 lines)

  ## Verdict
  Winner: [Answer 1 | Answer 2]
  Key Strength of Winner: <one sentence>
  Key Weakness of Loser:  <one sentence>
  Overall Reasoning:      <2-3 sentences>
"""

from __future__ import annotations
from agents.base_agent import BaseAgent
from agents.rubric import format_rubric_block, format_output_template, SCORING_GUIDE


QWEN35_MODEL = "qwen3.5:9b"
COMPASSJUDGER_MODEL = "hf.co/mradermacher/CompassJudger-1-7B-Instruct-GGUF:Q4_K_M"


def _get_auditor_model(requested: str = QWEN35_MODEL) -> str:
    """
    Returns requested model if available in Ollama, else falls back.

    Switched from CompassJudger-1 to qwen3.5:9b (2026-08-18) after discovering
    CompassJudger-1 gives all 8 rubric criteria the EXACT same score (e.g.
    9/9/9/9/9/9/9/9) on 75-80% of answers once scored blind (see RESEARCH_LOG.md
    Section 12-13) - i.e. it mostly wasn't discriminating between criteria at
    all, just writing boilerplate justifications around one flat number.
    qwen3.5:9b (with reasoning/thinking disabled - see create_judge_agent)
    showed 0% flat scores across 20 real Q&A pairs, with genuine score spread
    (50-77/80) and real per-answer differentiation - see RESEARCH_LOG.md
    Section 14. Before qwen3.5, CompassJudger-1 itself replaced qwen2.5 after
    an N=40 validation (RESEARCH_LOG.md Section 9-10) showing qwen2.5's own
    positivity bias (ceiling-hugging at 75-80/80) - that finding still stands,
    qwen2.5 is not being reconsidered here, only CompassJudger-1's suitability.
    """
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        if any("qwen3.5" in m for m in models):
            return QWEN35_MODEL
        if any("CompassJudger" in m for m in models):
            return COMPASSJUDGER_MODEL
        if any("qwen2.5" in m for m in models):
            return "qwen2.5:latest"
    except Exception:
        pass
    return "mistral"


def create_judge_agent(model_name: str | None = None) -> BaseAgent:
    """
    Returns an agent that scores and compares both LLM responses across
    8 explainable criteria with justified scores, forming an independent
    judgment rather than deferring to the Verifier's numbers.

    Role: Independent Answer Evaluator & Judge
    Model: qwen3.5:9b, a hybrid-reasoning model, run with thinking DISABLED
    (reasoning=False). With thinking left on, this model burns its entire
    token budget on internal chain-of-thought and returns an empty response
    for a prompt this long (confirmed directly - see RESEARCH_LOG.md Section
    14). With thinking off it produces genuinely differentiated per-criterion
    scores, unlike CompassJudger-1 which it replaced (see
    agents/judge_agent.py:_get_auditor_model).
    """
    actual_model = model_name or _get_auditor_model()
    is_qwen35 = "qwen3.5" in actual_model
    return BaseAgent(
        role="Answer Evaluator & Judge",
        model_name=actual_model,
        temperature=0.1,      # Very low → deterministic, consistent scoring
        num_predict=900,      # 8 criteria x 2 answers + verdict
        reasoning=(False if is_qwen35 else None),
        system_prompt=(
            "You are an impartial, highly analytical AI evaluator. "
            "You evaluate answers on EIGHT criteria, each scored 0-10. "
            "For EVERY score you give, you MUST write a one-sentence justification "
            "referencing specific content from the answer. "
            "Do NOT give a score without explaining it. "
            "You are scoring BLIND - no other evaluator's scores are shown to you. Score purely from "
            "your own reading of the question and the two answers.\n\n"
            "Ties are strictly forbidden. You must select exactly one winner (either Answer 1 or Answer 2). "
            "If the answers are very close, find a deciding factor (e.g. better examples, fewer factual errors, or clearer structure) to pick a single winner. "
            "Your final winner declaration MUST be exactly 'Winner: Answer 1' or 'Winner: Answer 2'. Never declare a tie or write multiple winners.\n\n"
            f"{SCORING_GUIDE}\n\n"
            "Criteria definitions:\n"
            f"{format_rubric_block()}\n\n"
            "Always use EXACTLY this format — no deviations:\n"
            f"{format_output_template('Answer 1')}\n\n"
            f"{format_output_template('Answer 2')}\n\n"
            "## Verdict\n"
            "Winner: Answer 1 or Answer 2\n"
            "Key Strength of Winner: <one sentence>\n"
            "Key Weakness of Loser: <one sentence>\n"
            "Overall Reasoning: <2–3 sentences>"
        ),
    )


def build_judge_prompt(query: str, answer1: str, answer2: str) -> str:
    """
    Build the explainable judge prompt.

    Deliberately does NOT include the Verifier's scores. Earlier versions
    passed the Verifier's raw output (including its per-criterion numbers)
    into this prompt "as a 3rd-party opinion" - in production this caused
    the Judge to reproduce the Verifier's totals digit-for-digit in ~86%
    of rows (32/37 in the first CompassJudger-1 v3-rubric batch), which
    means verifier_judge_agreement was mostly measuring anchoring, not
    genuine independent agreement. The Judge now scores blind from the
    question and the two answers only; agreement is computed afterward by
    comparing this output to the Verifier's separately-generated scores.
    See RESEARCH_LOG.md Section 12.

    Args:
        query:   The original user question.
        answer1: Whichever answer was randomly placed in position 1 by the
                 caller (may be LLaMA 3 or Mistral - deliberately not
                 revealed here, see module docstring / Section 17).
        answer2: The other answer, placed in position 2.

    Returns:
        A structured prompt string for the judge agent.
    """
    return (
        f"Evaluate the two AI-generated answers below using your 8-criterion rubric.\n\n"
        f"QUESTION: {query}\n\n"
        f"--- ANSWER 1 ---\n{answer1}\n\n"
        f"--- ANSWER 2 ---\n{answer2}\n\n"
        f"Score both answers on all 8 criteria with your OWN independent judgment. "
        f"For each score, write one sentence explaining exactly why, "
        f"referencing specific content from the answer. "
        f"Follow the exact output format from your instructions. "
        f"Compute TOTAL as sum of all 8 scores (max 80). "
        f"Then provide the Verdict section."
    )
