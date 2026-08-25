"""
agents/combiner_agent.py
------------------------
The Combiner Agent synthesises the best elements from both LLM responses
into a single, superior final answer.

Responsibilities:
  - Preserve unique insights from both answers
  - Remove contradictions or errors flagged by the Verifier
  - Produce a well-structured, clear, and comprehensive final answer
  - Ensure no critical information from either answer is lost
"""

from __future__ import annotations
from agents.base_agent import BaseAgent


def create_combiner_agent() -> BaseAgent:
    """
    Returns an agent that merges both LLM answers into a single
    refined, high-quality response.

    Role: Answer Synthesiser & Editor
    Model: llama3 (strong at summarisation and coherent writing)
    """
    return BaseAgent(
        role="Answer Synthesiser & Editor",
        model_name="llama3",
        temperature=0.4,      # Moderate temperature for fluent, creative synthesis
        num_predict=1000,
        system_prompt=(
            "You are a senior AI editor and synthesiser. You excel at taking "
            "multiple imperfect pieces of writing and crafting a single masterpiece. "
            "You never introduce new information not found in the source answers — "
            "your role is to merge and polish, not invent. The final answer you "
            "produce is presented directly to the end user, so it must be accurate, "
            "clear, and complete. Use well-structured paragraphs or bullet points."
        ),
    )


def build_combiner_prompt(
    query: str,
    answer1: str,
    answer2: str,
    verification: str,
    judgement: str,
) -> str:
    """
    Build the synthesis prompt with all available context.

    Args:
        query:        The original user question.
        answer1:      Whichever answer was in position 1 for this question.
        answer2:      Whichever answer was in position 2 for this question.
        verification: Verifier agent's feedback.
        judgement:    Judge agent's scores and winner decision.

    Returns:
        A structured prompt string for the combiner agent.
    """
    return (
        f"Synthesise a FINAL, DEFINITIVE answer to the question below by merging "
        f"the best parts of both AI responses. Use the verifier feedback to remove "
        f"any incorrect information, and the judge's decision as guidance.\n\n"
        f"QUESTION: {query}\n\n"
        f"--- ANSWER 1 ---\n{answer1}\n\n"
        f"--- ANSWER 2 ---\n{answer2}\n\n"
        f"--- VERIFIER FEEDBACK ---\n{verification}\n\n"
        f"--- JUDGE EVALUATION ---\n{judgement}\n\n"
        f"Your final answer must:\n"
        f"  1. Be more complete than either individual answer\n"
        f"  2. Contain no factual errors (use verifier feedback)\n"
        f"  3. Be well-structured and clearly written\n"
        f"  4. Directly and fully address the original question\n\n"
        f"  5. should contain at most 150 words strictly\n"
        f"Write the final combined answer now:"
    )
