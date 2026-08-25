# 🚀 How to Start the Agentic Multi-LLM Validation System

## Every Time You Want to Run the Project

Follow these steps **in order**, each in its own terminal / command prompt window.

---

## Step 1 — Start Ollama (Terminal 1)

```bash
ollama serve
```

> Keep this terminal **open and running** in the background.
> Ollama must be running before the web server starts.

---

## Step 2 — Start the Web Server (Terminal 2)

**On Windows (Command Prompt or PowerShell):**
```cmd
cd agentic-llm-system
venv\Scripts\activate
python server.py
```

**On macOS / Linux:**
```bash
cd agentic-llm-system
source venv/bin/activate
python server.py
```

You should see:
```
║   Agentic Multi-LLM Validation System  — Web Server          ║
║   Open: http://localhost:8000                                 ║
```

> Keep this terminal **open and running** too.

---

## Step 3 — Open the Web UI

Open your browser and go to:

```
http://localhost:8000
```

Type your question and click **🚀 Analyse with AI Agents**.

---

## To Stop the Project

- Press `Ctrl + C` in Terminal 2 to stop the web server.
- Press `Ctrl + C` in Terminal 1 to stop Ollama (optional).

---

## One-Time Setup (only needed if starting fresh on Windows or a new machine)

These steps are for setting up the project from scratch on a new machine.

### On Windows:
```cmd
# 1. Install Ollama from https://ollama.com/download

# 2. Pull the AI models (~9 GB total, run in Command Prompt/PowerShell)
ollama pull llama3
ollama pull mistral

# 3. Create Python virtual environment
cd agentic-llm-system
python -m venv venv

# 4. Activate virtual environment
venv\Scripts\activate

# 5. Install Python packages
pip install -r requirements.txt
```

### On macOS / Linux:
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the AI models (~9 GB total)
ollama pull llama3
ollama pull mistral

# 3. Create Python virtual environment
cd agentic-llm-system
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate

# 5. Install Python packages
pip install -r requirements.txt
```

---

## Quick Reference

| What                  | Windows Command                               | macOS / Linux Command                        |
|-----------------------|-----------------------------------------------|----------------------------------------------|
| Start Ollama          | `ollama serve`                                | `ollama serve`                               |
| Activate Python env   | `venv\Scripts\activate`                       | `source venv/bin/activate`                   |
| Start web server      | `python server.py`                            | `python server.py`                           |
| Open UI               | http://localhost:8000                         | http://localhost:8000                        |
| Run from terminal     | `python main.py "Your question here"`         | `python main.py "Your question here"`        |
| Check Ollama models   | `ollama list`                                 | `ollama list`                                |
| Pull a model          | `ollama pull llama3` / `ollama pull mistral`  | `ollama pull llama3` / `ollama pull mistral` |

