"""
main.py
-------
Agentic Multi-LLM Validation System
=====================================
Custom orchestrator for the full agentic pipeline (no external framework needed):

  Stage 1: LLaMA 3 agent    → generates Answer 1
  Stage 2: Mistral agent     → generates Answer 2
  Stage 3: Verifier agent    → fact-checks both answers
  Stage 4: Similarity scorer → embedding-based scores (offline, fast)
  Stage 5: Judge agent       → LLM-based scores + winner selection
  Stage 6: Combiner agent    → synthesises the definitive final answer

Tech stack:
  - LangChain + langchain-ollama  (LLM calls)
  - sentence-transformers          (semantic scoring)
  - rich                           (beautiful terminal output)
  - Ollama                         (local model serving — must be running)

Usage:
    python main.py
    python main.py "Your question here"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# ── Agent factories ─────────────────────────────────────────────────────────
from agents.llm_agents import create_llama3_agent, create_mistral_agent
from agents.verifier_agent import create_verifier_agent, build_verifier_prompt
from agents.judge_agent import create_judge_agent, build_judge_prompt
from agents.combiner_agent import create_combiner_agent, build_combiner_prompt

# ── Similarity / scoring utility ─────────────────────────────────────────────
from utils.similarity import score_response

# ── Rich console (global, used everywhere) ────────────────────────────────────
console = Console()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PipelineResult:
    """Holds all outputs produced by the pipeline stages."""
    query:        str
    answer1:      str          # LLaMA 3 response
    answer2:      str          # Mistral response
    verification: str          # Verifier feedback
    scores1:      dict         # Embedding-based score for answer 1
    scores2:      dict         # Embedding-based score for answer 2
    judgement:    str          # Judge evaluation + winner
    final_answer: str          # Combiner's synthesised answer
    elapsed_secs: float        # Total wall-clock time


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def _panel(title: str, content: str, style: str = "cyan") -> None:
    """Render a rich Panel with word-wrapped content."""
    console.print(Panel(content.strip(), title=f"[bold]{title}[/bold]",
                        border_style=style, expand=True))


def _score_table(scores1: dict, scores2: dict) -> None:
    """Render a formatted score comparison table."""
    table = Table(box=box.ROUNDED, border_style="dim", show_header=True,
                  header_style="bold magenta")
    table.add_column("Metric",              style="bold", width=26)
    table.add_column("LLM 1  (LLaMA 3)",   justify="center", width=22)
    table.add_column("LLM 2  (Mistral)",    justify="center", width=22)

    def _winner(v1: float, v2: float) -> tuple[str, str]:
        if v1 > v2:
            return f"[green]{v1}[/green]", str(v2)
        elif v2 > v1:
            return str(v1), f"[green]{v2}[/green]"
        return str(v1), str(v2)

    s1, s2 = _winner(scores1["semantic_similarity"], scores2["semantic_similarity"])
    table.add_row("Semantic Similarity", s1, s2)

    s1, s2 = _winner(scores1["length_score"], scores2["length_score"])
    table.add_row("Length / Completeness", s1, s2)

    table.add_row("Word Count",
                  str(scores1["word_count"]), str(scores2["word_count"]))

    s1, s2 = _winner(scores1["composite_score"], scores2["composite_score"])
    table.add_row("[bold]Composite Score (/10)[/bold]",
                  f"[bold]{s1}[/bold]", f"[bold]{s2}[/bold]")

    console.print(table)

    winner_label = (
        "[green]LLM 1 (LLaMA 3)[/green]"
        if scores1["composite_score"] >= scores2["composite_score"]
        else "[green]LLM 2 (Mistral)[/green]"
    )
    console.print(f"\n  ► Embedding-score winner: {winner_label}\n")


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_stage(label: str, fn) -> str:
    """Run a pipeline stage with a progress spinner and timing."""
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]{label}...[/bold cyan]"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("", total=None)
        t0 = time.time()
        result: str = fn()
        elapsed = time.time() - t0

    console.print(f"  [dim]✓ {label} completed in {elapsed:.1f}s[/dim]")
    return result


def run_pipeline(query: str) -> PipelineResult:
    """
    Execute the full multi-agent validation pipeline.

    Stages:
      1. LLaMA 3   → Answer 1
      2. Mistral   → Answer 2
      3. Verifier  → Fact-check report
      4. Scorer    → Embedding-based similarity scores
      5. Judge     → Rubric-based evaluation + winner
      6. Combiner  → Final synthesised answer

    Args:
        query: The user's input question.

    Returns:
        A PipelineResult populated with all stage outputs.
    """
    t_start = time.time()

    console.print(Rule("[bold yellow]Initialising Agents[/bold yellow]"))
    llama3_agent   = create_llama3_agent()
    mistral_agent  = create_mistral_agent()
    verifier_agent = create_verifier_agent()
    judge_agent    = create_judge_agent()
    combiner_agent = create_combiner_agent()
    console.print("  [dim]All 5 agents ready.[/dim]\n")

    # ── Stage 1: LLaMA 3 answer ──────────────────────────────────────────────
    console.print(Rule("[bold cyan]Stage 1 — LLaMA 3 Answer Generator[/bold cyan]"))
    answer1 = run_stage(
        "LLaMA 3 generating answer",
        lambda: llama3_agent.run(
            f"Answer the following question thoroughly and accurately.\n\n"
            f"Question: {query}\n\n"
            f"Provide a well-structured, comprehensive answer of at most 350 words. "
            f"Use clear language, headings, or bullet points where appropriate."
        ),
    )

    # ── Stage 2: Mistral answer ──────────────────────────────────────────────
    console.print(Rule("[bold cyan]Stage 2 — Mistral Answer Generator[/bold cyan]"))
    answer2 = run_stage(
        "Mistral generating answer",
        lambda: mistral_agent.run(
            f"Answer the following question thoroughly and accurately.\n\n"
            f"Question: {query}\n\n"
            f"Provide a well-structured, comprehensive answer of at most 350 words. "
            f"Include practical examples and unique insights."
        ),
    )

    # ── Stage 3: Verifier ────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Stage 3 — Fact-Checker & Verifier[/bold cyan]"))
    verifier_prompt = build_verifier_prompt(query, answer1, answer2)
    verification = run_stage(
        "Verifier analysing both answers",
        lambda: verifier_agent.run(verifier_prompt),
    )

    # ── Stage 4: Similarity scoring (offline) ────────────────────────────────
    console.print(Rule("[bold cyan]Stage 4 — Embedding Similarity Scorer[/bold cyan]"))
    console.print("  [dim]Computing semantic embeddings...[/dim]")
    scores1 = score_response(query, answer1)
    scores2 = score_response(query, answer2)
    console.print("  [dim]✓ Scoring complete.[/dim]\n")

    # ── Stage 5: Judge ───────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Stage 5 — Judge Evaluator[/bold cyan]"))
    judge_prompt = build_judge_prompt(query, answer1, answer2)
    judgement = run_stage(
        "Judge scoring and selecting winner",
        lambda: judge_agent.run(judge_prompt),
    )

    # ── Stage 6: Combiner ────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Stage 6 — Answer Synthesiser[/bold cyan]"))
    combiner_prompt = build_combiner_prompt(
        query, answer1, answer2, verification, judgement
    )
    final_answer = run_stage(
        "Combiner synthesising final answer",
        lambda: combiner_agent.run(combiner_prompt),
    )

    elapsed = time.time() - t_start
    return PipelineResult(
        query=query,
        answer1=answer1,
        answer2=answer2,
        verification=verification,
        scores1=scores1,
        scores2=scores2,
        judgement=judgement,
        final_answer=final_answer,
        elapsed_secs=elapsed,
    )


# ============================================================================
# OUTPUT RENDERER
# ============================================================================

def display_results(result: PipelineResult) -> None:
    """Render all pipeline outputs to the terminal using rich styling."""

    console.print(Rule("[bold yellow]Pipeline Results[/bold yellow]"))

    _panel("📝  ANSWER FROM LLM 1  (LLaMA 3)", result.answer1, style="blue")
    _panel("📝  ANSWER FROM LLM 2  (Mistral)", result.answer2, style="blue")
    _panel("🔍  VERIFIER FEEDBACK", result.verification, style="yellow")

    console.print(Panel(
        "[bold]Embedding-Based Similarity Scores[/bold]",
        border_style="magenta", expand=True
    ))
    _score_table(result.scores1, result.scores2)

    _panel("⚖️   JUDGE EVALUATION & VERDICT", result.judgement, style="magenta")
    _panel("✦   FINAL COMBINED ANSWER", result.final_answer, style="green")

    console.print(
        f"\n[dim]Total pipeline time: {result.elapsed_secs:.1f}s[/dim]\n"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    console.print(Panel(
        "[bold yellow]Agentic Multi-LLM Validation System[/bold yellow]  v2.0\n"
        "[dim]Powered by LangChain · Ollama · Sentence-Transformers[/dim]",
        border_style="yellow",
        expand=False,
    ))

    # Accept query from CLI argument or prompt interactively
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        console.print(f"[bold]Query:[/bold] {user_query}\n")
    else:
        user_query = console.input("[bold green]Enter your question:[/bold green] ").strip()

    if not user_query:
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    try:
        result = run_pipeline(user_query)
        display_results(result)
    except Exception as exc:
        console.print_exception()
        console.print(
            f"\n[red bold]Pipeline failed:[/red bold] {exc}\n"
            "[dim]Make sure Ollama is running: [bold]ollama serve[/bold]\n"
            "And models are pulled: [bold]ollama pull llama3 && ollama pull mistral[/bold][/dim]"
        )
        sys.exit(1)
