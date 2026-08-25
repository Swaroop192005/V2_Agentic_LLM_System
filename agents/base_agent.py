"""
agents/base_agent.py
--------------------
Base class for all agents in the Agentic Multi-LLM Validation System.

Each agent wraps a LangChain OllamaLLM and exposes a `run(prompt)` method.
This replaces CrewAI with a lightweight, dependency-free alternative that
works on Python 3.14 without build issues.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from langchain_ollama import OllamaLLM


@dataclass
class BaseAgent:
    """
    A simple agent backed by a local Ollama model via LangChain.

    Attributes:
        role       : Human-readable name for this agent (displayed in output).
        model_name : Ollama model to use (e.g. 'llama3', 'mistral').
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        num_predict: Max tokens to generate per response.
        system_prompt: Optional system instruction prepended to every call.
        reasoning  : For hybrid-reasoning models (e.g. qwen3.5) that support an
                     internal "thinking" phase - False disables it. Left as
                     None for models that don't support the flag (harmless).
                     Without this, a reasoning model can burn its entire
                     num_predict budget on internal thinking and return an
                     EMPTY response for long/complex prompts (confirmed with
                     qwen3.5:9b on the judge rubric prompt - see
                     RESEARCH_LOG.md Section 14).
    """
    role: str
    model_name: str
    temperature: float = 0.7
    num_predict: int = 512
    system_prompt: str = ""
    reasoning: bool | None = None
    _llm: OllamaLLM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            keep_alive="5m",
            timeout=120.0,
            reasoning=self.reasoning,
        )

    def run(self, prompt: str) -> str:
        """
        Invoke the agent with a prompt and return the LLM response.

        Args:
            prompt: The full prompt to send to the model.

        Returns:
            The model's text response as a string.
        """
        full_prompt = (
            f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
        )
        print(f"[{self.role}] Thinking...", flush=True)
        response: str = self._llm.invoke(full_prompt)
        return response.strip()
