"""
server.py
---------
FastAPI backend for the Agentic Multi-LLM Validation System.

Serves:
  GET  /              → frontend/index.html
  GET  /api/health    → Ollama connectivity check
  POST /api/stream    → SSE stream that runs the pipeline stage-by-stage
  GET  /api/context   → Returns the last 5 stored query metadata records

SSE Event types emitted to the browser:
  stage      – a stage just started      { stage, name, status:"running" }
  result     – a stage completed         { stage, name, output }
  scores     – similarity scores         { scores1, scores2 }
  confidence – weighted score for the winning answer after Stage 5
                                          { attempt, weighted_score, judge, similarity, agreement,
                                            verifier_judge_agreement, wikipedia, wikipedia_source,
                                            wikidata, wikidata_source, threshold, winner,
                                            total_a, total_b, verifier_total_a, verifier_total_b }
  regenerate – confidence was below threshold, retrying stages 1-5
                                          { attempt, reason, threshold }
  error      – something went wrong      { message }
  done       – pipeline finished         { elapsed }

Run:
    python server.py
    # or
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import random
import time
import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import ollama as ollama_client

from agents.llm_agents import create_llama3_agent, create_mistral_agent
from agents.verifier_agent import create_verifier_agent, build_verifier_prompt
from agents.judge_agent import create_judge_agent, build_judge_prompt
from agents.combiner_agent import create_combiner_agent, build_combiner_prompt
from utils.similarity import score_response
from utils.context_store import init_db, save_query, get_all_context_records, build_context_prefix, clear_context
from utils.rag_store import init_rag_db, retrieve_context, index_document, get_rag_stats
from utils.rubric_parser import parse_judge_scores, parse_rubric_scores, compute_agreement, remap_ab
from utils.scoring import compute_weighted_score, needs_regeneration, DEFAULT_THRESHOLD, MAX_REGENERATION_ATTEMPTS
from utils.eval_metrics import compute_bertscore_batch
from utils.fact_check import verify_against_wikipedia
from utils.wikidata_check import verify_against_wikidata

# ── App setup ────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup; clean up on shutdown."""
    # Startup: create SQLite context table & RAG vector store (idempotent)
    init_db()
    init_rag_db()
    print("[Context DB] SQLite context store initialised at context.db")
    print("[RAG Store] Local Vector Store initialised at data/rag_knowledge.db")
    yield
    # Shutdown: nothing to clean up for SQLite


app = FastAPI(title="Agentic Multi-LLM Validation System", version="2.0", lifespan=lifespan)

# Serve static assets (CSS, JS) from frontend/
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main frontend page."""
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/api/health")
async def health_check():
    """
    Check if Ollama is reachable and which models are pulled.
    Returns { ok: bool, models: [...] }
    """
    try:
        models_resp = ollama_client.list()
        names = [m.model for m in models_resp.models]
        return {
            "ok": True,
            "models": names,
            "has_llama3":  any("llama3"  in n for n in names),
            "has_mistral": any("mistral" in n for n in names),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "models": []}


@app.get("/api/context")
async def get_context():
    """
    Return the last 5 stored query metadata records from SQLite.
    Used by the frontend to display conversation history.
    Returns { count: int, records: [...] }
    """
    records = get_all_context_records()
    return {"count": len(records), "records": records}


@app.delete("/api/context")
async def clear_context_endpoint():
    """
    Delete all stored conversation-history records (the rolling last-5 queries).
    Use this to start a fresh conversation with no prior-topic context bleeding
    into new, unrelated questions.
    Returns { cleared: int }
    """
    deleted = await asyncio.get_running_loop().run_in_executor(None, clear_context)
    return {"cleared": deleted}


@app.get("/api/rag/stats")
async def rag_stats():
    """Return RAG vector store stats (doc count, chunks, filenames)."""
    stats = get_rag_stats()
    return {"ok": True, "stats": stats}


@app.post("/api/rag/upload")
async def rag_upload(request: Request, file: UploadFile | None = File(None), files: list[UploadFile] | None = File(None)):
    """
    Endpoint for document indexing. Accepts PDF, TXT, MD, etc.
    Supports single or multiple file uploads via multipart form or JSON body.
    """
    upload_list = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)
    
    if upload_list:
        total_chunks = 0
        processed_files = []
        for f in upload_list:
            fname = f.filename or "Uploaded_Doc.pdf"
            fbytes = await f.read()
            c_count = index_document(fname, fbytes)
            total_chunks += c_count
            processed_files.append(fname)
        return {"ok": True, "filename": ", ".join(processed_files), "chunks_created": total_chunks, "count": len(processed_files)}
    
    try:
        body = await request.json()
        filename = body.get("filename", "Uploaded_Doc.txt").strip()
        content = body.get("content", "").strip()
        if not content:
            return {"ok": False, "message": "Content is empty."}
        chunks_created = index_document(filename, content)
        return {"ok": True, "filename": filename, "chunks_created": chunks_created}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@app.post("/api/stream")
async def stream_pipeline(request: Request):
    """
    Accept { query: string, use_rag: bool } and stream SSE events as the pipeline progresses.
    Each agent stage emits events in real-time so the UI can update live.
    Context from rolling memory and optional RAG vector search are injected into LLM prompts.
    """
    body = await request.json()
    query: str = body.get("query", "").strip()
    use_rag: bool = body.get("use_rag", True)  # Enabled by default if RAG docs exist

    if not query:
        async def error_gen():
            yield {"event": "error", "data": json.dumps({"message": "Query is empty."})}
        return EventSourceResponse(error_gen())

    return EventSourceResponse(
        _run_pipeline_sse(query, use_rag=use_rag),
        media_type="text/event-stream",
    )


# ── Pipeline SSE generator ────────────────────────────────────────────────────

async def _run_pipeline_sse(query: str, use_rag: bool = True) -> AsyncIterator[dict]:
    """
    Run all pipeline stages and yield SSE events as each completes.
    Runs blocking LLM calls in a thread pool so the event loop stays free.
    Injects rolling context (last 5 queries) and RAG vector context into LLM prompts.
    """
    loop = asyncio.get_running_loop()
    t_start = time.time()

    def _emit(event: str, data: dict) -> dict:
        return {"event": event, "data": json.dumps(data)}

    try:
        # ── Build context prefix from last 5 queries (may be empty string) ──
        context_prefix = build_context_prefix(query)
        if context_prefix:
            yield _emit("context", {"loaded": True, "entries": len(get_all_context_records())})

        # ── RAG Vector Retrieval (Optional / Auto) ─────────────────────────
        rag_prompt_prefix = ""
        if use_rag:
            rag_chunks = await loop.run_in_executor(None, retrieve_context, query, 3)
            if rag_chunks:
                yield _emit("rag", {"retrieved": True, "count": len(rag_chunks), "chunks": rag_chunks})
                formatted_docs = "\n\n".join([f"[{c['doc_name']} (score: {c['similarity']})]:\n{c['content']}" for c in rag_chunks])
                rag_prompt_prefix = (
                    f"--- RETRIEVED GROUND-TRUTH CONTEXT DOCUMENTS (RAG) ---\n"
                    f"{formatted_docs}\n"
                    f"-------------------------------------------------------\n\n"
                    f"Use the verified context documents above to inform and ground your response accurately.\n\n"
                )

        # Initialise all agents (fast — just object creation)
        llama3_agent   = create_llama3_agent()
        mistral_agent  = create_mistral_agent()
        verifier_agent = create_verifier_agent()
        judge_agent    = create_judge_agent()
        combiner_agent = create_combiner_agent()

        llama3_prompt = (
            f"{rag_prompt_prefix}"
            f"{context_prefix}"
            f"Answer the following question thoroughly and accurately.\n\n"
            f"Question: {query}\n\n"
            f"Provide a well-structured, comprehensive answer of at least 150 words."
        )
        mistral_prompt = (
            f"{rag_prompt_prefix}"
            f"{context_prefix}"
            f"Answer the following question thoroughly and accurately.\n\n"
            f"Question: {query}\n\n"
            f"Provide a well-structured, comprehensive answer of at least 150 words. "
            f"Include practical examples and unique insights."
        )

        # ── Stages 1-5, with regeneration if the winning answer scores low ──
        # Each attempt regenerates BOTH answers from scratch (the judge/verifier
        # can't tell which one was weak in isolation - a low weighted score means
        # low confidence in the winner, so we retry the whole comparison).
        best_attempt = None   # highest-scoring attempt seen so far, in case we exhaust attempts
        previous_score = None  # the failing score from the prior attempt, shown in the retry message

        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 2):  # +1 initial + MAX retries
            if attempt == 1:
                yield _emit("stage", {"stage": 1, "name": "LLaMA 3", "status": "running"})
                yield _emit("stage", {"stage": 2, "name": "Mistral",  "status": "running"})
            else:
                yield _emit("regenerate", {
                    "attempt": attempt,
                    "reason": "low confidence score",
                    "previous_score": previous_score,
                    "threshold": DEFAULT_THRESHOLD,
                })
                yield _emit("stage", {"stage": 1, "name": "LLaMA 3", "status": "running", "attempt": attempt})
                yield _emit("stage", {"stage": 2, "name": "Mistral",  "status": "running", "attempt": attempt})

            answer1, answer2 = await asyncio.gather(
                loop.run_in_executor(None, llama3_agent.run, llama3_prompt),
                loop.run_in_executor(None, mistral_agent.run, mistral_prompt),
            )
            yield _emit("result", {"stage": 1, "name": "LLaMA 3", "output": answer1})
            yield _emit("result", {"stage": 2, "name": "Mistral",  "output": answer2})

            # Randomize which answer the Verifier/Judge see as "Answer 1" vs
            # "Answer 2" for THIS attempt, independent of the fixed UI display
            # order (LLaMA 3 always shown as Answer 1 above) - avoids the
            # position/identity bias found in the batch dataset (see
            # RESEARCH_LOG.md Section 17). Both agents' prompts already never
            # reveal brand identity. Their parsed results are remapped back to
            # UI order (remap_ab) immediately after parsing, so every line
            # below this block is unchanged and still operates on
            # answer1=LLaMA3 / answer2=Mistral as before.
            swap = random.random() < 0.5
            judge_view_1, judge_view_2 = (answer2, answer1) if swap else (answer1, answer2)

            # ── Stage 3 + 4: Verifier & Similarity IN PARALLEL ───────────────
            yield _emit("stage", {"stage": 3, "name": "Verifier",           "status": "running"})
            yield _emit("stage", {"stage": 4, "name": "Similarity Scorer",  "status": "running"})

            verifier_prompt = build_verifier_prompt(query, judge_view_1, judge_view_2, rag_context=rag_prompt_prefix)

            (verification, (scores1, scores2)) = await asyncio.gather(
                loop.run_in_executor(None, verifier_agent.run, verifier_prompt),
                asyncio.gather(
                    loop.run_in_executor(None, score_response, query, answer1),
                    loop.run_in_executor(None, score_response, query, answer2),
                ),
            )
            yield _emit("result", {"stage": 3, "name": "Verifier", "output": verification})
            yield _emit("scores", {"scores1": scores1, "scores2": scores2})

            # ── Stage 5: Judge ────────────────────────────────────────────────
            yield _emit("stage", {"stage": 5, "name": "Judge", "status": "running"})
            judge_prompt = build_judge_prompt(query, judge_view_1, judge_view_2)
            judgement = await loop.run_in_executor(None, judge_agent.run, judge_prompt)
            yield _emit("result", {"stage": 5, "name": "Judge", "output": judgement})

            # ── Weighted confidence score for the winning answer ────────────
            # Both Verifier and Judge scored all 8 rubric factors independently -
            # parse both, remap back to UI order (a=LLaMA3, b=Mistral), and use
            # their agreement as an extra signal.
            verifier_parsed = remap_ab(parse_rubric_scores(verification), swap)
            judge_parsed = remap_ab(parse_judge_scores(judgement, query, judge_view_1, judge_view_2), swap)
            winner = judge_parsed["winner"]
            total_winner = judge_parsed["total_a"] if winner == "A" else judge_parsed["total_b"]
            sim_winner = scores1["semantic_similarity"] if winner == "A" else scores2["semantic_similarity"]
            verifier_judge_agreement = compute_agreement(verifier_parsed, judge_parsed)

            winner_answer = answer1 if winner == "A" else answer2

            async def _safe_model_agreement():
                try:
                    bs = await loop.run_in_executor(None, compute_bertscore_batch, [answer1], [answer2])
                    return bs["f1"][0]
                except Exception:
                    return None  # optional; scoring still works without it

            async def _safe_wikipedia():
                try:
                    return await loop.run_in_executor(None, verify_against_wikipedia, query, winner_answer)
                except Exception:
                    return None  # optional; network issues shouldn't break the pipeline

            async def _safe_wikidata():
                try:
                    return await loop.run_in_executor(None, verify_against_wikidata, query, winner_answer)
                except Exception:
                    return None  # optional; network issues shouldn't break the pipeline

            model_agreement, wikipedia_result, wikidata_result = await asyncio.gather(
                _safe_model_agreement(), _safe_wikipedia(), _safe_wikidata()
            )
            wikipedia_score = wikipedia_result["score"] if wikipedia_result else None
            wikidata_score = wikidata_result["score"] if wikidata_result else None

            breakdown = compute_weighted_score(
                judge_total=total_winner,
                similarity_0_1=sim_winner,
                model_agreement_0_1=model_agreement,
                verifier_judge_agreement_0_1=verifier_judge_agreement,
                wikipedia_0_1=wikipedia_score,
                wikidata_0_1=wikidata_score,
            )
            yield _emit("confidence", {
                "attempt": attempt,
                "weighted_score": breakdown.weighted_score,
                "judge": breakdown.judge,
                "similarity": breakdown.similarity,
                "agreement": breakdown.agreement,
                "verifier_judge_agreement": verifier_judge_agreement,
                "wikipedia": wikipedia_score,
                "wikipedia_source": wikipedia_result["source_title"] if wikipedia_result else None,
                "wikidata": wikidata_score,
                "wikidata_source": wikidata_result["source_title"] if wikidata_result else None,
                "threshold": DEFAULT_THRESHOLD,
                "winner": winner,
                "total_a": judge_parsed["total_a"],
                "total_b": judge_parsed["total_b"],
                "verifier_total_a": verifier_parsed["total_a"],
                "verifier_total_b": verifier_parsed["total_b"],
            })

            if best_attempt is None or breakdown.weighted_score > best_attempt["score"]:
                best_attempt = {
                    "score": breakdown.weighted_score,
                    "answer1": answer1, "answer2": answer2,
                    "verification": verification, "judgement": judgement,
                }

            if not needs_regeneration(breakdown, DEFAULT_THRESHOLD) or attempt > MAX_REGENERATION_ATTEMPTS:
                break
            previous_score = breakdown.weighted_score  # carried into next loop's "regenerate" event

        # Use whichever attempt scored highest (usually the last one, unless a retry regressed)
        answer1, answer2 = best_attempt["answer1"], best_attempt["answer2"]
        verification, judgement = best_attempt["verification"], best_attempt["judgement"]

        # ── Stage 6: Combiner ────────────────────────────────────────────────
        yield _emit("stage", {"stage": 6, "name": "Combiner", "status": "running"})
        combiner_prompt = build_combiner_prompt(query, answer1, answer2, verification, judgement)
        final_answer = await loop.run_in_executor(None, combiner_agent.run, combiner_prompt)
        yield _emit("result", {"stage": 6, "name": "Combiner", "output": final_answer})

        # ── Done ─────────────────────────────────────────────────────────
        elapsed = round(time.time() - t_start, 1)

        # ── Persist query + answer summary to the rolling context DB ─────
        await loop.run_in_executor(None, save_query, query, final_answer)

        yield _emit("done", {"elapsed": elapsed})

    except Exception as exc:
        yield _emit("error", {
            "message": str(exc),
            "detail": traceback.format_exc(),
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n+--------------------------------------------------------------+")
    print("|   Agentic Multi-LLM Validation System  -- Web Server          |")
    print("|   Open: http://localhost:8000                                 |")
    print("+--------------------------------------------------------------+\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
