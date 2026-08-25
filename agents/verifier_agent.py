"""
agents/verifier_agent.py
------------------------
The Verifier Agent independently scores both LLM responses on the shared
8-factor rubric (agents/rubric.py) — the SAME rubric the Judge uses.

This replaces the old free-form "hallucination narrative" design: previously
the Judge simply read the Verifier's prose and trusted it, making Judge
scores (especially Factual Accuracy) effectively dependent on phi3's
unstructured opinion. Now phi3 and qwen2.5 each independently produce
structured per-factor scores, so their agreement/disagreement becomes a
measurable signal instead of an invisible dependency.

The Verifier does NOT declare a winner - that stays the Judge's unique role.

Like the Judge, the Verifier is never told which physical model (LLaMA 3 or
Mistral) produced "Answer 1" vs "Answer 2" - the caller randomizes position
per question specifically to prevent position/identity bias. See
RESEARCH_LOG.md Section 17.
"""

from __future__ import annotations
from agents.base_agent import BaseAgent
from agents.rubric import format_rubric_block, format_output_template, SCORING_GUIDE


def _get_verifier_model(requested: str = "phi3:latest") -> str:
    """Returns requested model if available in Ollama, else safe fallback."""
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        if any("phi3" in m for m in models):
            return "phi3:latest"
        elif any("gemma" in m for m in models):
            return "gemma:latest"
    except Exception:
        pass
    return "llama3"


def create_verifier_agent(model_name: str | None = None) -> BaseAgent:
    """
    Returns an agent that independently scores both LLM-generated answers
    on the shared 8-factor rubric.

    Role: Independent Fact-Checker & Structured Evaluator
    Model: phi3:latest (Microsoft AI - 3rd-party independent verifier)
    """
    actual_model = model_name or _get_verifier_model()
    return BaseAgent(
        role="Fact-Checker & Structured Evaluator",
        model_name=actual_model,
        temperature=0.1,      # Very low temperature for consistent judgement
        num_predict=700,      # 8 criteria x 2 answers needs more room than the old prose format
        system_prompt=(
            "You are an impartial 3rd-party expert fact-checking agent with deep knowledge across "
            "science, technology, engineering, and current events. You independently score both "
            "answers on EIGHT criteria, each 0-10, exactly like an evaluator would - you do NOT "
            "declare a winner (that is a separate step). "
            "For EVERY score you give, you MUST write a one-sentence justification referencing "
            "specific content from the answer. Do NOT give a score without explaining it. "
            "You do NOT rewrite the answers - you only critique them against factual knowledge "
            "and any reference documents provided.\n\n"
            f"{SCORING_GUIDE}\n\n"
            "Criteria definitions:\n"
            f"{format_rubric_block()}\n\n"
            "Always use EXACTLY this format — no deviations:\n"
            f"{format_output_template('Answer 1')}\n\n"
            f"{format_output_template('Answer 2')}"
        ),
    )


def build_verifier_prompt(query: str, answer1: str, answer2: str, rag_context: str = "") -> str:
    """
    Build the verification prompt combining the query, both answers, and optional RAG ground-truth context.
    """
    context_str = f"--- RETRIEVED REFERENCE GROUND-TRUTH DOCUMENTS ---\n{rag_context}\n\n" if rag_context else ""

    return (
        f"{context_str}"
        f"You are given two AI-generated answers to the following question:\n\n"
        f"QUESTION: {query}\n\n"
        f"--- ANSWER 1 ---\n{answer1}\n\n"
        f"--- ANSWER 2 ---\n{answer2}\n\n"
        f"Score both answers on all 8 criteria. For each score, write one sentence explaining "
        f"exactly why, referencing specific content from the answer"
        f"{' and the reference documents' if rag_context else ''}. "
        f"If no reference documents were provided, score Context Relevance based on whether the "
        f"answer stays on-topic for the question. "
        f"Follow the exact output format from your instructions. "
        f"Compute TOTAL as the sum of all 8 scores (max 80)."
    )
