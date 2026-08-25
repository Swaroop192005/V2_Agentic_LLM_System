# 🤖 Agentic Multi-LLM Validation System
### Complete Technical Guide — For Teammates

> **What this document covers:** Every technical concept, design decision, line-level explanation, and architectural choice in this project. By the end you should be able to understand, run, extend, or debug any part of the system — even if you've never seen it before.

---

## Table of Contents

1. [The Big Picture — What Is This?](#1-the-big-picture)
2. [Why Multi-Agent? The Core Idea](#2-why-multi-agent)
3. [Tech Stack — Every Library Explained](#3-tech-stack)
4. [Project File Structure](#4-project-file-structure)
5. [How to Run the Project](#5-how-to-run-the-project)
6. [Architecture Deep Dive](#6-architecture-deep-dive)
7. [The Pipeline — Stage by Stage](#7-the-pipeline-stage-by-stage)
8. [Agent Design — Every Agent Explained](#8-agent-design)
9. [The Similarity Scorer (Non-LLM)](#9-the-similarity-scorer)
10. [Context Memory System (SQLite)](#10-context-memory-system)
11. [FastAPI Server & SSE Streaming](#11-fastapi-server--sse-streaming)
12. [Frontend — Real-time UI Explained](#12-frontend)
13. [Async Parallelism — How We Halve the Wait Time](#13-async-parallelism)
14. [The Training Data Generator](#14-the-training-data-generator)
15. [Live Dataset Viewer](#15-live-dataset-viewer)
16. [Known Issues & Fixes](#16-known-issues--fixes)
17. [Key Design Decisions & Why](#17-key-design-decisions)
18. [Current Status & Next Steps](#18-current-status--next-steps)
19. [Glossary](#19-glossary)

---

## 1. The Big Picture

### What Does This System Do?

You type a question. Instead of one AI giving you one answer, **six specialized AI agents** kick in:

1. **LLaMA 3** generates Answer A
2. **Mistral** generates Answer B (independently, in parallel)
3. **Verifier** fact-checks both answers for hallucinations
4. **Scorer** measures how semantically relevant each answer is (no LLM involved)
5. **Judge** evaluates both answers on 5 criteria and declares a winner
6. **Combiner** synthesizes the best parts of both into a single final answer ≤150 words

The result is consistently better than any single LLM could produce alone.

### Why Is This Interesting?

- **100% offline** — runs on your laptop, no OpenAI API, no internet required
- **Open source models only** — LLaMA 3 (Meta) + Mistral (Mistral AI) via Ollama
- **Real-time streaming** — you see each agent's output as it finishes, not all at once
- **Self-improving** — it's generating its own training dataset (150+ samples so far) to eventually replace the slow LLM judge with a fast custom model

---

## 2. Why Multi-Agent?

### The Problem With Single LLMs

Any single LLM has:
- **Training biases** — blind spots in certain domains
- **Hallucinations** — confident wrong answers with no self-correction
- **No competition** — nothing pushes it to produce better output

### The Solution: Competition + Validation

```
LLaMA 3 answer  ──┐
                  ├── Verifier checks both ──── Judge picks winner ──── Combiner merges
Mistral answer  ──┘
```

By making two *different* models answer the same question:
- Their errors don't overlap (one tends to catch what the other misses)
- The Judge acts as a third independent opinion
- The Combiner ensures no good information is lost

This is similar to **ensemble learning** in machine learning — combining weak learners into a stronger one.

---

## 3. Tech Stack

Every library used, what it does, and why it was chosen:

### Core AI / LLM

| Library | Version | What It Does | Why We Use It |
|---|---|---|---|
| **Ollama** | Latest | Runs LLaMA 3 and Mistral locally as HTTP API | Only tool that makes running 7B+ models locally trivially easy |
| **LangChain** | ≥0.3.0 | Orchestration framework for LLMs | Clean abstraction for prompt → LLM → response |
| **langchain-ollama** | ≥0.2.0 | Bridge between LangChain and Ollama | Official integration, replaces deprecated community wrapper |
| **ollama** (Python client) | ≥0.2.0 | Direct Ollama API access (for health checks) | Used separately to list available models |

### Embeddings & Scoring

| Library | Version | What It Does | Why We Use It |
|---|---|---|---|
| **sentence-transformers** | ≥2.7.0 | Converts text to 384-dimensional vectors | Enables computing semantic similarity without an LLM |
| **scikit-learn** | ≥1.4.0 | Provides `cosine_similarity` function | Fast, battle-tested implementation |
| **numpy** | ≥1.26.0 | Array math, clipping similarity scores | Required by both sentence-transformers and sklearn |

### Web Server

| Library | Version | What It Does | Why We Use It |
|---|---|---|---|
| **FastAPI** | ≥0.111.0 | The HTTP API framework | Async-first, auto OpenAPI docs, perfect for SSE |
| **Uvicorn** | ≥0.29.0 | ASGI server that runs FastAPI | Only production-grade async Python web server |
| **sse-starlette** | ≥1.6.5 | Adds Server-Sent Events support to FastAPI | Simplest way to stream real-time events without WebSockets |

### Utilities

| Library | Version | What It Does |
|---|---|---|
| **pydantic** | ≥2.7.0 | Data validation (used internally by FastAPI) |
| **rich** | ≥13.0.0 | Beautiful terminal output with progress bars (used in training generator) |
| **python-dotenv** | ≥1.0.0 | Load `.env` config files if needed |

### Built-in Python

| Module | Used For |
|---|---|
| `sqlite3` | Context store + training data database |
| `asyncio` | Parallel execution of pipeline stages |
| `json` | SSE event serialization |
| `pathlib` | Cross-platform file paths |
| `dataclasses` | Agent class definition |
| `textwrap` | Truncating text summaries |
| `datetime` | Timestamps in context store |

---

## 4. Project File Structure

```
agentic-llm-system/
│
├── 📄 server.py                    # FastAPI backend — THE main entry point
├── 📄 main.py                      # Terminal-only runner (no web UI)
├── 📄 generate_training_data.py    # Batch generator for 1000 training samples
├── 📄 viewer_server.py             # Lightweight server for the live data viewer
├── 📄 requirements.txt             # All Python dependencies
├── 📄 README.md                    # Quick setup guide
├── 📄 STARTUP.md                   # One-liner startup commands
│
├── 📁 agents/                      # ← All 5 AI agents live here
│   ├── __init__.py
│   ├── base_agent.py               # Parent class all agents inherit from
│   ├── llm_agents.py               # LLaMA 3 + Mistral agent factories
│   ├── verifier_agent.py           # Fact-checker agent
│   ├── judge_agent.py              # Scoring rubric + verdict agent
│   └── combiner_agent.py           # Final answer synthesizer
│
├── 📁 utils/                       # ← Utility modules
│   ├── __init__.py
│   ├── similarity.py               # Embedding-based semantic scorer
│   └── context_store.py            # SQLite rolling memory (last 5 queries)
│
├── 📁 frontend/                    # ← Web interface (pure HTML/CSS/JS)
│   ├── index.html                  # The single page app
│   ├── style.css                   # Dark glassmorphism styles + animations
│   └── app.js                      # Real-time SSE client + pipeline visualizer
│
├── 📁 data/
│   └── judge_training.db           # SQLite DB with 150+ training samples
│
├── 📄 context.db                   # SQLite DB for rolling query context
├── 📄 training_log.txt             # Background generator stdout log
├── 📄 server_log.txt               # FastAPI server log
├── 📄 data_viewer_live.html        # Live dashboard UI (served by viewer_server.py)
└── 📄 data_viewer.html             # Static snapshot viewer (standalone)
```

---

## 5. How to Run the Project

### Prerequisites

```bash
# 1. Install Ollama (macOS)
brew install ollama

# 2. Start Ollama daemon
ollama serve

# 3. Pull the two models (one-time, ~4GB each)
ollama pull llama3
ollama pull mistral

# 4. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
python server.py
# → Open http://localhost:8000
```

### Run the Training Data Generator (background)

```bash
nohup python generate_training_data.py > training_log.txt 2>&1 &
# Runs silently, logs progress to training_log.txt
# Check progress:
tail -f training_log.txt
```

### Run the Live Data Viewer

```bash
python viewer_server.py
# → Opens http://localhost:7788 automatically
# → Auto-refreshes every 5 seconds as new training rows arrive
```

---

## 6. Architecture Deep Dive

### The Full Data Flow

```
Browser
  │
  │  POST /api/stream  { query: "..." }
  ▼
FastAPI (server.py)
  │
  │  asynccontextmanager lifespan → init_db() on startup
  │
  ├─ build_context_prefix(query)     ← pulls last 5 queries from context.db
  │
  ├─ asyncio.gather() ──────── PARALLEL ──────────────────────────────────┐
  │   ├─ loop.run_in_executor(llama3_agent.run, prompt)                   │
  │   └─ loop.run_in_executor(mistral_agent.run, prompt)                  │
  │                                                                        │
  │  [SSE: stage events + result events emitted to browser]               │
  │                                                                        │
  ├─ asyncio.gather() ──────── PARALLEL ──────────────────────────────────┤
  │   ├─ loop.run_in_executor(verifier_agent.run, verifier_prompt)        │
  │   └─ asyncio.gather()                                                  │
  │       ├─ loop.run_in_executor(score_response, query, answer1)         │
  │       └─ loop.run_in_executor(score_response, query, answer2)         │
  │                                                                        │
  ├─ loop.run_in_executor(judge_agent.run, judge_prompt)  ← SEQUENTIAL   │
  │                                                                        │
  ├─ loop.run_in_executor(combiner_agent.run, combiner_prompt) ← SEQ     │
  │                                                                        │
  ├─ save_query(query, final_answer)  ← persist to context.db            │
  │                                                                        │
  └─ yield SSE "done" event  { elapsed: X.X }  ──────────────────────────┘
```

### Why `run_in_executor`?

LLM calls are **blocking** — they make synchronous HTTP calls to the Ollama daemon and wait. FastAPI runs on an `asyncio` event loop, which must *never* be blocked.

`loop.run_in_executor(None, fn, arg)` offloads the blocking call to a **thread pool**, letting the event loop continue handling SSE keep-alive pings and other requests while the LLM thinks.

```python
# This would BLOCK the event loop — WRONG:
answer = llama3_agent.run(prompt)

# This runs in a thread, yields control to event loop — CORRECT:
answer = await loop.run_in_executor(None, llama3_agent.run, prompt)
```

---

## 7. The Pipeline — Stage by Stage

### Stage 1 & 2 — Parallel Answer Generation

Both models receive the **same question** plus the **last 5 queries as context**. They run *simultaneously* via `asyncio.gather()`.

**LLaMA 3 prompt template:**
```
[CONVERSATION CONTEXT — last queries the user asked]
  Q1: <oldest query>
  A1 (summary): <answer summary>
  ...
[Use the above context ONLY to understand the user's topic and intent...]

Answer the following question thoroughly and accurately.

Question: <user query>

Provide a well-structured, comprehensive answer of at least 150 words.
```

**Mistral prompt** is identical but adds: *"Include practical examples and unique insights."*

**Why different prompts?** We deliberately bias each model slightly differently to maximize **answer diversity**. If both get the same prompt, they might produce nearly identical answers.

**Temperature = 0.7** for both — this is the "creativity dial":
- `0.0` = most deterministic (same output every time)
- `1.0` = most random/creative
- `0.7` = good balance of consistency and variety

**num_predict = 600** — max tokens to generate (~450 words)

---

### Stage 3 — Verifier Agent

Runs **in parallel with** the Scorer (Stage 4).

The Verifier gets:
- The original question
- Answer 1 (LLaMA 3)
- Answer 2 (Mistral)

It produces structured output under two headings:
```
[Answer 1 Verification]
<critique of Answer 1>

[Answer 2 Verification]
<critique of Answer 2>
```

**Temperature = 0.1** — extremely low because we want **consistent, deterministic fact-checking**, not creative hallucination about hallucinations.

**num_predict = 450** — shorter cap since critique doesn't need lengthy output.

**Model: LLaMA 3** — used here because it has strong instruction-following at low temperature.

---

### Stage 4 — Similarity Scorer (No LLM)

Runs in parallel with the Verifier. **This is NOT an LLM** — it's a pure math computation.

**How it works:**
```python
# 1. Load the all-MiniLM-L6-v2 model (cached as singleton)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Encode both texts as 384-dimensional vectors
embeddings = model.encode([query, response])

# 3. Compute cosine similarity (how "aligned" the vectors are)
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
# Result: 0.0 (unrelated) to 1.0 (same meaning)

# 4. Compute length score (penalizes short answers)
word_count = len(response.split())
length_score = clip(word_count / 150.0, 0.0, 1.0)
# 150+ words = score of 1.0

# 5. Composite weighted score (out of 10)
composite = (0.70 * semantic_sim + 0.30 * length_score) * 10
```

**Why 70/30 weighting?** Semantic similarity (relevance) is more important than raw length. A short, accurate answer beats a long, rambling one. But completely one-word answers should be penalized.

**The `all-MiniLM-L6-v2` model:**
- 22.7M parameters (tiny — loads in milliseconds)
- Produces 384-dimensional sentence embeddings
- Trained specifically for semantic similarity tasks
- Runs fully locally — no API call

---

### Stage 5 — Judge Agent

The Judge receives **everything**: both answers + the Verifier's feedback.

**Evaluation rubric (5 criteria, 0–10 each):**

| Criterion | What Is Measured |
|---|---|
| **Factual Accuracy** | Are claims correct? Any hallucinations? Uses Verifier feedback here |
| **Completeness** | Does it cover all key aspects of the question? |
| **Clarity** | Is it well-structured, logical, easy to read? |
| **Relevance** | Does it directly answer the question without padding? |
| **Depth of Reasoning** | Does it explain *why*, not just *what*? |

**Scoring guide given to the model:**
```
0–3: Poor    — major problems, misleading, or incomplete
4–6: Average — correct but missing depth or has minor issues
7–9: Good    — strong answer with minor room for improvement
10:  Perfect — comprehensive, accurate, clear, and insightful
```

**Every score requires a one-sentence justification.** This is enforced in the system prompt:
> *"For EVERY score you give, you MUST write a one-sentence justification referencing specific content from the answer."*

**Output format (strictly enforced):**
```
## Answer 1 (LLaMA 3)
- Factual Accuracy:   8/10 | <justification>
- Completeness:       7/10 | <justification>
- Clarity:            9/10 | <justification>
- Relevance:          8/10 | <justification>
- Depth of Reasoning: 7/10 | <justification>
- TOTAL: 39/50

## Answer 2 (Mistral)
...
- TOTAL: 41/50

## Verdict
Winner: Answer 2
Key Strength of Winner: <one sentence>
Key Weakness of Loser:  <one sentence>
Overall Reasoning:      <2–3 sentences>
```

**Temperature = 0.1** — near-deterministic for consistent scoring.
**Model: Mistral** — chosen for its strong analytical capability and structured output.

---

### Stage 6 — Combiner Agent

The Combiner sees **everything** — both answers, verifier feedback, and the judge's verdict.

**Its strict constraints:**
1. Be more complete than either individual answer
2. Contain no factual errors (use verifier feedback to filter)
3. Be well-structured (paragraphs or bullets)
4. Directly address the original question
5. **≤150 words strictly** — enforced via prompt instruction

**Temperature = 0.4** — moderate, allowing fluent creative writing but not too random.
**Model: LLaMA 3** — chosen specifically for its superior summarization and writing quality.
**num_predict = 1000** — given the most token budget since it produces the final user-facing answer.

---

## 8. Agent Design

### The `BaseAgent` Class

All 5 agents inherit from a single `BaseAgent` dataclass in `agents/base_agent.py`:

```python
@dataclass
class BaseAgent:
    role: str           # Human-readable name (e.g., "Judge")
    model_name: str     # Ollama model (e.g., "llama3", "mistral")
    temperature: float  # 0.0 = deterministic, 1.0 = creative
    num_predict: int    # Max tokens to generate
    system_prompt: str  # Instructions prepended to every call
    _llm: OllamaLLM     # LangChain wrapper (created in __post_init__)
```

**The `run()` method:**
```python
def run(self, prompt: str) -> str:
    full_prompt = f"{self.system_prompt}\n\n{prompt}"
    response = self._llm.invoke(full_prompt)
    return response.strip()
```

It just concatenates the system prompt with the user prompt and calls `OllamaLLM.invoke()`. Simple and transparent.

### Why Not Use CrewAI or AutoGen?

We explicitly **replaced CrewAI** with this lightweight custom BaseAgent. Reasons:
- CrewAI had **Python 3.14 build issues** (Pydantic v1 conflicts)
- We didn't need CrewAI's task planning — our pipeline is sequential/parallel in a known fixed order
- Less dependency = fewer failure points
- The BaseAgent is 58 lines of code and does exactly what we need

### Agent Specifications Summary

| Agent | File | Model | Temp | Max Tokens | Role |
|---|---|---|---|---|---|
| LLaMA 3 Generator | `llm_agents.py` | llama3 | 0.7 | 600 | Primary answer |
| Mistral Generator | `llm_agents.py` | mistral | 0.7 | 600 | Alternative answer |
| Verifier | `verifier_agent.py` | llama3 | 0.1 | 450 | Fact-check both answers |
| Judge | `judge_agent.py` | mistral | 0.1 | 700 | Score & pick winner |
| Combiner | `combiner_agent.py` | llama3 | 0.4 | 1000 | Synthesize final answer |

---

## 9. The Similarity Scorer

Located in `utils/similarity.py`. This is a **pure NLP computation, no LLM required**.

### The Singleton Pattern

The `all-MiniLM-L6-v2` sentence transformer model takes ~1–2 seconds to load. We use a module-level global variable to load it **only once** and reuse it for every call:

```python
_model: SentenceTransformer | None = None   # starts as None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:                      # only load on first call
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model                           # return cached model on all future calls
```

### Environment Variables — Crash Prevention

At the top of `similarity.py`, we set 5 environment variables **before** importing sentence-transformers:

```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
```

**Why?** Python 3.14 on macOS has a bug where forking a process inside an `asyncio` event loop causes a **SIGABRT crash**. The `sentence-transformers` library uses `joblib` internally, which tries to spawn worker processes. These env variables force everything into single-threaded mode, preventing the crash.

---

## 10. Context Memory System

Located in `utils/context_store.py`. Uses **SQLite** — Python's built-in database.

### Database Schema

```sql
CREATE TABLE IF NOT EXISTS query_context (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,   -- ISO-8601 datetime
    query          TEXT    NOT NULL,   -- The user's question
    answer_summary TEXT,               -- First 400 chars of final answer
    word_count     INTEGER NOT NULL DEFAULT 0
);
```

### WAL Mode

```python
conn.execute("PRAGMA journal_mode=WAL;")
```

**WAL = Write-Ahead Logging** — a SQLite mode that allows:
- Multiple readers while one writer is active
- No read locks blocking writes
- Better concurrency for the async server + background generator running simultaneously

### The Rolling Window

After every save, we immediately prune:

```python
conn.execute("""
    DELETE FROM query_context
    WHERE id NOT IN (
        SELECT id FROM query_context
        ORDER BY id DESC
        LIMIT 5        ← keep only latest 5
    )
""")
```

This means the DB never grows beyond 5 rows. It's a rolling context window, not an ever-growing log.

### How Context Is Injected Into Prompts

`build_context_prefix()` reads the last 5 records and formats them:

```
[CONVERSATION CONTEXT — last queries the user asked]
  Q1: What is quantum entanglement?
  A1 (summary): Quantum entanglement is a phenomenon where two particles...
  Q2: How does it differ from classical correlation?
  A2 (summary): Unlike classical correlation, entanglement is non-local...

[Use the above context ONLY to understand the user's topic and intent.
Do not repeat or rehash previous answers. Focus on the NEW question below.]
```

This prefix is prepended to **both** LLaMA 3 and Mistral prompts, making them context-aware across conversations.

---

## 11. FastAPI Server & SSE Streaming

### What Is Server-Sent Events (SSE)?

SSE is a protocol where the server **pushes** data to the browser over a single long-lived HTTP connection. Unlike WebSockets, SSE is:
- **One-directional** (server → client only)
- **HTTP/1.1 compatible** (no upgrade handshake needed)
- **Auto-reconnecting** (browser reconnects if connection drops)
- **Simpler** to implement than WebSockets for our use case

**SSE wire format:**
```
event: stage
data: {"stage": 1, "name": "LLaMA 3", "status": "running"}

event: result
data: {"stage": 1, "name": "LLaMA 3", "output": "Quantum entanglement is..."}

event: done
data: {"elapsed": 312.4}
```

### The SSE Event Types We Emit

| Event | When | Data |
|---|---|---|
| `context` | If prior context exists | `{ loaded: true, entries: N }` |
| `stage` | When a stage starts | `{ stage, name, status: "running" }` |
| `result` | When a stage finishes | `{ stage, name, output: "..." }` |
| `scores` | After Scorer finishes | `{ scores1: {...}, scores2: {...} }` |
| `error` | On any exception | `{ message, detail }` |
| `done` | Pipeline complete | `{ elapsed: X.X }` |

### API Routes

| Method | Route | What It Does |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `GET` | `/api/health` | Returns Ollama status + available models |
| `POST` | `/api/stream` | Accepts `{ query }`, returns SSE stream |
| `GET` | `/api/context` | Returns last 5 context records as JSON |
| Static | `/static/*` | Serves all files in `frontend/` directory |

### Lifespan Context Manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()   # Create context.db table on startup (idempotent)
    yield       # App runs here
    # cleanup on shutdown (nothing needed for SQLite)
```

This is FastAPI's recommended pattern for startup/shutdown logic (replaces deprecated `@app.on_event`).

---

## 12. Frontend

Located in `frontend/` — pure HTML, CSS, JavaScript, no framework.

### `index.html` — Structure

- Health indicator (green/red dot showing Ollama status)
- Query text area with character counter + Cmd+Enter shortcut
- Pipeline visualization section (the animated nodes)
- Results section (expandable accordion for each stage)
- "View Context" slide-in panel

### `style.css` — Design System

**Dark glassmorphism** design:
- Background: `#0a0e1a` (deep navy)
- Cards: `rgba(255,255,255,0.05)` with `backdrop-filter: blur(20px)`
- Accent: `#00d4ff` (cyan) for active state
- LLaMA color: `#4f8ef7` (blue)
- Mistral color: `#a78bfa` (purple)
- Success: `#34d399` (green)

Animations:
- **Pulsing glow rings** on active nodes using `@keyframes pulse`
- **Connector arrows** that animate cyan only when both upstream nodes complete
- **Slide-in** for the context panel

### `app.js` — Real-time Pipeline Tracker

**SSE Client setup:**
```javascript
const eventSource = new EventSource('/api/stream', {
    method: 'POST',   // not directly supported — uses fetch + ReadableStream
    body: JSON.stringify({ query })
});
```

*Note: Because POST SSE isn't natively supported by the EventSource API, the frontend uses `fetch()` with a `ReadableStream` reader to simulate it.*

**Event handlers:**
```javascript
on('stage',  data) → mark node as "running" + start pulse animation
on('result', data) → populate result accordion + mark node "done"
on('scores', data) → render score comparison table
on('done',   data) → show elapsed time + "Ask New Question" button
on('error',  data) → show error message with details
```

**Pipeline node states:**
- `waiting` — grey circle, no animation
- `running` — cyan glow pulse animation
- `done` — filled circle, green checkmark
- `error` — red circle

**Connector logic:** Each connector between parallel groups only activates (turns cyan and animates the arrow) when **both** nodes in the upstream group are marked `done`. This accurately represents the data dependency — the Verifier/Scorer can't start until both LLaMA 3 and Mistral finish.

---

## 13. Async Parallelism

This is the key performance optimization. Without it, the pipeline would be ~10 minutes. With it, it's ~5–6 minutes.

### How Python asyncio Works

Python is single-threaded. `asyncio` gives the illusion of concurrency by switching between tasks when one is waiting (e.g., for I/O).

```
Time →
Thread 1: [LLaMA 3 waiting for Ollama response...................done]
Thread 2:        [Mistral waiting for Ollama response................done]
                                                                    ↓
                                                             Both ready → proceed
```

### The Two Parallel Stages

**Stage 1+2 (Generator parallel):**
```python
answer1, answer2 = await asyncio.gather(
    loop.run_in_executor(None, llama3_agent.run, llama3_prompt),
    loop.run_in_executor(None, mistral_agent.run, mistral_prompt),
)
```

**Stage 3+4 (Verifier + Scorer parallel):**
```python
(verification, (scores1, scores2)) = await asyncio.gather(
    loop.run_in_executor(None, verifier_agent.run, verifier_prompt),
    asyncio.gather(
        loop.run_in_executor(None, score_response, query, answer1),
        loop.run_in_executor(None, score_response, query, answer2),
    ),
)
```

Note: The Scorer itself is parallelized — both `score_response(query, answer1)` and `score_response(query, answer2)` run concurrently.

### Why `run_in_executor` and not just `asyncio.gather`?

`asyncio.gather` works with **coroutines** (async functions). `agent.run()` is a **synchronous** function (it makes a blocking HTTP call to Ollama). You can't directly `await` a sync function.

`run_in_executor` submits the sync function to a **ThreadPoolExecutor** and returns a future that asyncio can await. This bridges sync → async without blocking the event loop.

---

## 14. The Training Data Generator

Located in `generate_training_data.py`. This is a **standalone background script** — not part of the web server.

### Purpose

Train a custom judge model that can replace the slow Mistral LLM judge. To train it, we need labeled data: questions + both answers + scores for each answer.

### The 1000 Questions

Spread across 10 domains (100 questions each):
- Science & Technology
- History & Geography
- Health & Medicine
- Economics & Finance
- Philosophy & Ethics
- Environment & Climate
- Psychology & Behaviour
- Society & Culture
- Mathematics & Logic
- Space & Astronomy

The diversity ensures the custom judge won't overfit to any one topic.

### Database Schema (judge_training.db)

```sql
CREATE TABLE training_samples (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    question         TEXT,
    answer_a         TEXT,     -- LLaMA 3's answer
    answer_b         TEXT,     -- Mistral's answer
    verifier_text    TEXT,     -- Verifier's full output
    judge_raw        TEXT,     -- Judge's full output (raw text)
    sem_sim_a        REAL,     -- Cosine similarity of answer_a to question
    sem_sim_b        REAL,     -- Cosine similarity of answer_b to question
    length_score_a   REAL,     -- Normalized length score for answer_a
    length_score_b   REAL,     -- Normalized length score for answer_b
    composite_a      REAL,     -- (0.7 * sem_sim + 0.3 * length) * 10 for A
    composite_b      REAL,     -- Same for B
    word_count_a     INTEGER,
    word_count_b     INTEGER,
    factual_a        INTEGER,  -- Judge score 0-10
    completeness_a   INTEGER,
    clarity_a        INTEGER,
    relevance_a      INTEGER,
    depth_a          INTEGER,
    total_a          INTEGER,  -- Sum of above 5 (max 50)
    factual_b        INTEGER,
    completeness_b   INTEGER,
    clarity_b        INTEGER,
    relevance_b      INTEGER,
    depth_b          INTEGER,
    total_b          INTEGER,
    winner           TEXT,     -- "A", "B", or "unknown"
    elapsed_secs     REAL,     -- How long this sample took to generate
    created_at       TEXT      -- ISO timestamp
);
```

### Score Parsing

The generator uses regex to extract individual scores from the Judge's text output:

```python
# Parse scores like "- Factual Accuracy:   8/10 | justification"
pattern = r"Factual Accuracy[:\s]+(\d+)/10"
match = re.search(pattern, judge_raw, re.IGNORECASE)
factual_a = int(match.group(1)) if match else 0
```

### Resumability

Before processing each question, the generator checks if it's already done:

```python
existing = conn.execute(
    "SELECT id FROM training_samples WHERE question = ?", (question,)
).fetchone()
if existing:
    continue   # skip — already processed
```

This means if the script crashes or is stopped, it picks up exactly where it left off.

### Running Time

Each question takes **~5–7 minutes** (LLaMA 3 + Mistral generation + Verifier + Judge + Combiner, two parallel stages).

1000 questions × 6 min = **~100 hours total** (run unattended over several days/nights).

**Current status: 150 samples collected** (~15% complete).

---

## 15. Live Dataset Viewer

Two files work together:

### `viewer_server.py`

A minimal Python HTTP server that:
- Serves `data_viewer_live.html` at `/`
- Exposes `/api/all` — all rows from `judge_training.db` as JSON
- Exposes `/api/data?since_id=N` — only rows newer than ID N (incremental polling)
- Exposes `/api/stats` — aggregate statistics

```bash
python viewer_server.py
# → http://localhost:7788
```

### `data_viewer_live.html`

A single-page dashboard that:
- **On load:** fetches all records from `/api/all`
- **Every 5 seconds:** fetches `/api/data?since_id=MAX_KNOWN_ID` to find new rows only
- **On new rows:** flashes them green in the table, shows a toast notification
- Thin blue **progress bar** at top counts down to next refresh
- **Filter** by winner (LLaMA / Mistral / Tie)
- **Search** by question text
- **Sort** by score, composite, or generation time
- **Click any row** → opens full modal with:
  - Per-metric score bars for both models
  - Side-by-side answer text
  - Raw judge verdict + verifier analysis
  - All metadata (semantic similarity, word counts, elapsed time)

---

## 16. Known Issues & Fixes

### SIGABRT Crash on Python 3.14 + macOS

**Symptom:** Server crashes with `SIGABRT` after the first request.
**Cause:** `sentence-transformers` uses `joblib` which tries to fork worker processes inside the asyncio event loop — not safe on Python 3.14 + macOS.
**Fix:** Set these env vars **before** importing sentence-transformers (done in `similarity.py`):
```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
```

### LangChain Pydantic v1 Deprecation Warning

**Symptom:** `PydanticDeprecatedSince20` warning in console.
**Cause:** Some LangChain internals still use Pydantic v1 patterns.
**Status:** Cosmetic only — no functional impact. Fixed in newer LangChain releases.

### Ollama Single-Threaded Limitation

**Problem:** Ollama can only run one model inference at a time on most hardware. When the background generator is running, starting the web server causes `generate_training_data.py` to slow down significantly.
**Mitigation:** Stop the generator before using the web app. Resume it after.

### Judge Output Parsing Can Fail

If the Judge produces malformed output (missing `TOTAL: XX/50`), the regex parser falls back to `0` for that score. This is logged. The row is still saved with whatever scores could be extracted.

---

## 17. Key Design Decisions

| Decision | Rationale |
|---|---|
| **No external APIs** | 100% offline — no costs, no data leakage, works without internet |
| **Two different LLMs** | Model diversity reduces blind spots. LLaMA 3 and Mistral have complementary strengths |
| **Mistral for judging** | Strong analytical, structured output at low temperature. More consistent than LLaMA at scoring |
| **LLaMA 3 for combining** | Superior creative writing, better at fluent summarization |
| **asyncio parallelism** | Cuts pipeline time from ~10 min to ~5–6 min (40% reduction) |
| **SSE over WebSocket** | One-way streaming only needed. SSE is simpler, HTTP/1.1 native, auto-reconnecting |
| **SQLite over Redis/Postgres** | Zero infrastructure, zero config, persists across restarts. More than sufficient at this scale |
| **`all-MiniLM-L6-v2` embedding** | Lightweight (22M params), fast, accurate enough for relevance scoring. No API call needed |
| **Custom BaseAgent vs CrewAI** | CrewAI had Python 3.14 incompatibilities. Our 58-line BaseAgent does the job cleanly |
| **Structured judge output** | Forcing a specific format makes parsing reliable and makes scores auditable |
| **Rolling 5-query context** | Enough for conversational coherence; too many would clutter the prompt |
| **WAL mode SQLite** | Allows concurrent reads during writes — important when generator + server run simultaneously |

---

## 18. Current Status & Next Steps

### What's Done ✅

- [x] Full 6-agent pipeline working end-to-end
- [x] Real-time SSE streaming to browser
- [x] Parallel execution (Stages 1+2, 3+4)
- [x] Context-aware memory (rolling 5-query SQLite store)
- [x] Similarity scoring (embedding-based, no LLM)
- [x] Judge with explainable rubric scoring
- [x] Premium dark glassmorphism frontend
- [x] Training data generator running in background
- [x] 150+ labeled training samples collected
- [x] Live dataset viewer with real-time updates

### In Progress 🔄

- [ ] **Dataset collection** — need 1000 samples total (~850 remaining, ~6 days)
- [ ] **Custom judge model training** — BERT/RoBERTa regressor on the 5 criteria

### Planned 📋

- [ ] Human feedback buttons (👍/👎) in the UI to augment training data
- [ ] More models via Ollama (Gemma 2, Phi-3, DeepSeek)
- [ ] Advanced context search panel
- [ ] Export pipeline results as PDF

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Agent** | An AI entity with a specific role, model, and instructions |
| **asyncio** | Python's built-in async concurrency library |
| **ASGI** | Asynchronous Server Gateway Interface — the standard FastAPI/Uvicorn use |
| **Cosine Similarity** | Measure of angle between two vectors (0 = unrelated, 1 = identical meaning) |
| **Embedding** | A fixed-size numerical representation of text's semantic meaning |
| **Hallucination** | When an LLM confidently states something false as if it were fact |
| **LangChain** | Framework that wraps LLMs with a standard interface |
| **Lifespan** | FastAPI's startup/shutdown lifecycle hook |
| **num_predict** | Ollama parameter: max number of tokens (roughly words) to generate |
| **Ollama** | Tool that runs LLMs locally as an HTTP API |
| **run_in_executor** | asyncio method to run blocking sync code in a thread pool |
| **SSE** | Server-Sent Events — HTTP streaming from server to browser |
| **Temperature** | LLM randomness control: 0 = deterministic, 1 = very creative |
| **WAL** | Write-Ahead Logging — SQLite concurrency mode |
| **sentence-transformers** | Library for converting text to semantic embedding vectors |
| **singleton** | Design pattern where only one instance of an object exists (reused model) |
| **ThreadPoolExecutor** | Pool of background threads for running sync code alongside asyncio |

---

*Document written: April 2026*
*Project: Agentic Multi-LLM Validation System v2.0*
*Stack: Python 3.14 · LangChain · Ollama · LLaMA 3 · Mistral · FastAPI · SQLite · sentence-transformers*
