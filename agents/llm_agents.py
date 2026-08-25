"""
agents/llm_agents.py
--------------------
Defines two primary LLM agents powered by different local Ollama models:

  - Agent 1  →  llama3
  - Agent 2  →  mistral

Each agent generates an independent answer for the same user query.
Using two different models increases answer diversity and reduces the risk
of a single model's blind-spots going undetected.
"""

from __future__ import annotations
from agents.base_agent import BaseAgent


def create_llama3_agent() -> BaseAgent:
    """
    Returns an agent backed by the local llama3 Ollama model.

    Role     : Primary Answer Generator (LLaMA 3)
    Goal     : Produce a thorough, accurate answer for the given query.
    """
    return BaseAgent(
        role="LLama 3 Answer Generator",
        model_name="llama3",
        temperature=0.7,
        num_predict=600,
        system_prompt=(
            "You are an advanced AI assistant powered by Meta's LLaMA 3 model. "
            "Your goal is to provide comprehensive, well-structured, and factually "
            "accurate answers. Use clear language, logical organisation, and include "
            "examples or details where relevant. Always verify your reasoning before "
            "presenting an answer."
        ),
    )


def create_mistral_agent() -> BaseAgent:
    """
    Returns an agent backed by the local mistral Ollama model.

    Role     : Secondary Answer Generator (Mistral)
    Goal     : Produce an alternative, complementary answer for the same query.
    """
    return BaseAgent(
        role="Mistral Answer Generator",
        model_name="mistral",
        temperature=0.7,
        num_predict=600,
        system_prompt=(
            "You are an AI assistant powered by Mistral AI's open-source model. "
            "You are known for efficient reasoning, structured outputs, and strong "
            "performance on analytical and factual tasks. Provide a well-reasoned, "
            "concise, and factually accurate answer. Include practical examples or "
            "insights that complement other AI systems' answers."
        ),
    )
