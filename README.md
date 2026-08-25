# Agentic Multi-LLM Validation System

> An agentic AI pipeline where multiple local LLMs collaborate to **generate**, **validate**, **score**, and **synthesise** answers — powered by CrewAI, Ollama, and Sentence-Transformers.

---

## System Architecture

```
User Query
    │
    ├──► Task 1: LLaMA 3 Agent      → Answer 1
    ├──► Task 2: Mistral Agent       → Answer 2
    │
    ├──► Task 3: Verifier Agent      → Fact-check & hallucination report
    │
    ├──► [Offline] Similarity Scorer → Embedding-based scores (0–10)
    │
    ├──► Task 4: Judge Agent         → LLM-based scores + winner selection
    │
    └──► Task 5: Combiner Agent      → Final synthesised answer
```

---

## Project Structure

```
agentic-llm-system/
│
├── main.py                    # Orchestrator — builds and runs the full pipeline
│
├── agents/
│   ├── __init__.py
│   ├── llm_agents.py          # LLaMA 3 + Mistral answer-generation agents
│   ├── verifier_agent.py      # Fact-checker & hallucination detector
│   ├── judge_agent.py         # Evaluator — scores + selects winner
│   └── combiner_agent.py      # Answer synthesiser & editor
│
├── utils/
│   ├── __init__.py
│   └── similarity.py          # Embedding-based cosine similarity scoring
│
└── requirements.txt           # All Python dependencies
```

---

## Prerequisites

### 1. Install Ollama (macOS / Linux / Windows)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**  
Download the installer from [https://ollama.com/download](https://ollama.com/download)

---

### 2. Start the Ollama server

```bash
ollama serve
```

> Keep this terminal open. Ollama must be running for the agents to work.

---

### 3. Pull the required models

In a **new terminal**:

```bash
# Pull LLaMA 3 (~4.7 GB)
ollama pull llama3

# Pull Mistral (~4.1 GB)
ollama pull mistral
```

Verify they are installed:
```bash
ollama list
```

You should see both `llama3` and `mistral` in the output.

---

### 4. Set up Python environment

```bash
# Navigate to the project directory
cd agentic-llm-system

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** The first run will also download the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB) automatically.

---

## Running the System

### Interactive mode (prompts for input):
```bash
python main.py
```

### Inline query mode:
```bash
python main.py "What is the difference between supervised and unsupervised learning?"
```

### Example queries to try:
```bash
python main.py "Explain how transformers work in natural language processing."

python main.py "What are the main causes of climate change?"

python main.py "How does the TCP/IP protocol stack work?"

python main.py "Explain SOLID principles in software engineering."
```

---

## Sample Output

```
======================================================================
  AGENTIC MULTI-LLM VALIDATION SYSTEM
======================================================================
  Query: What is the difference between supervised and unsupervised learning?

======================================================================
  ANSWER FROM LLM 1  (LLaMA 3)
======================================================================
  Supervised learning is a type of machine learning where the model is
  trained on labeled data...

======================================================================
  ANSWER FROM LLM 2  (Mistral)
======================================================================
  The key distinction between supervised and unsupervised learning lies
  in the availability of labeled training data...

======================================================================
  VERIFIER FEEDBACK
======================================================================
  [Answer 1 Verification]
  ✓ No major hallucinations detected...

  [Answer 2 Verification]
  ✓ Accurate description of clustering algorithms...

======================================================================
  EMBEDDING-BASED SIMILARITY SCORES
======================================================================
  LLM 1 (LLaMA 3):
    Semantic Similarity : 0.8342
    Length Score        : 0.9800  (147 words)
    ── Composite Score  : 8.78 / 10

  LLM 2 (Mistral):
    Semantic Similarity : 0.8109
    Length Score        : 0.8600  (129 words)
    ── Composite Score  : 8.24 / 10

  ► Similarity winner: LLM 1 (LLaMA 3)  [8.78/10]

======================================================================
  JUDGE EVALUATION & VERDICT
======================================================================
  Answer 1 — Accuracy: 9/10, Completeness: 8/10, Clarity: 9/10, Total: 8.7/10
  Answer 2 — Accuracy: 8/10, Completeness: 7/10, Clarity: 8/10, Total: 7.7/10
  Winner: Answer 1
  Reason: Answer 1 provides a more thorough explanation...

======================================================================
  ✦ FINAL COMBINED ANSWER
======================================================================
  Supervised and unsupervised learning represent two fundamental
  paradigms in machine learning...
```

---

## How It Works

### Agent Pipeline (Sequential)

| Step | Agent | Model | Purpose |
|------|-------|-------|---------|
| 1 | LLaMA 3 Agent | `llama3` | Generates the first complete answer |
| 2 | Mistral Agent | `mistral` | Generates an alternative answer |
| 3 | Verifier Agent | `llama3` | Fact-checks both answers (hallucination detection) |
| 4 | Judge Agent | `mistral` | Scores each answer (accuracy, completeness, clarity) |
| 5 | Combiner Agent | `llama3` | Synthesises the best final answer |

### Scoring (Offline — runs between Tasks 3 and 4)

| Metric | Weight | Description |
|--------|--------|-------------|
| Semantic Similarity | 70% | Cosine similarity between query and response embeddings |
| Length / Completeness | 30% | Normalised word-count score (150 words = 1.0) |
| **Composite Score** | — | Weighted blend, expressed out of 10 |

---

## Configuration & Extension

### Change LLM models
Edit `agents/llm_agents.py` and modify the model name in `_make_ollama_llm()`:
```python
return Ollama(model="your-model-name", ...)
```

Any model available via `ollama list` can be used.

### Adjust scoring weights
Edit `utils/similarity.py`:
```python
composite = (0.70 * semantic_sim + 0.30 * length_score) * 10
#             ^^^^                   ^^^^
#         semantic weight         length weight
```

### Add a new agent
1. Create a new file in `agents/` following the pattern of existing agents.
2. Import and instantiate the agent in `main.py`.
3. Add a new `Task` to the `build_tasks()` function.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` / Ollama error | Run `ollama serve` in a separate terminal |
| `model not found` error | Run `ollama pull llama3` and `ollama pull mistral` |
| Slow responses | Normal — local LLMs are CPU/GPU bound. Use a machine with 16+ GB RAM |
| `ModuleNotFoundError` | Activate venv and re-run `pip install -r requirements.txt` |
| CUDA / GPU acceleration | Install the CUDA version of Ollama for faster inference |

---

## Academic Use

This system demonstrates:
- **Multi-agent collaboration** using CrewAI's sequential process
- **Local LLM inference** without any paid API (100% offline capable)
- **Hallucination detection** via a dedicated verifier agent
- **Dual scoring mechanism** (semantic embedding + LLM judge)
- **Answer fusion** as the final synthesis step

Suitable for final-year projects in AI, NLP, and distributed AI systems.

---

## License

MIT License — free to use, modify, and distribute.
