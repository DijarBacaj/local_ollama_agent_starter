# Local Ollama Agent Starter

It is designed for a machine with **8 GB VRAM + 16 GB RAM**.

## Best default model 

Start with:

```bash
ollama pull qwen2.5-coder:7b
```

You can also try:

```bash
ollama pull llama3.1:8b
```

Then set the model in `.env`.

---

## What this project can do

The agent can:

- chat locally through Ollama
- list files in a workspace folder
- read text files
- write text files
- do math
- save durable notes
- search saved notes
- search Wikipedia
- search public Hugging Face models
- optionally run short Python snippets inside a workspace sandbox

## Project structure

```text
local_ollama_agent_starter/
├─ app.py
├─ agent.py
├─ config.py
├─ memory.py
├─ tools.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ install_models.bat
├─ run_app.bat
├─ memory/
└─ workspace/
```

---

## Step-by-step setup on Windows

### 1) Install Ollama

Install Ollama on Windows, then open PowerShell or CMD and run:

```bash
ollama --version
```

### 2) Pull a model

```bash
ollama pull qwen2.5-coder:7b
```

### 3) Create and activate a virtual environment

In the project folder:

```bash
py -3 -m venv .venv
.venv\Scripts\activate
```

### 4) Install Python packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5) Create your `.env`

Copy `.env.example` to `.env`.

PowerShell:

```powershell
Copy-Item .env.example .env
```

CMD:

```cmd
copy .env.example .env
```

### 6) Start the app

```bash
python app.py
```

Open the local Gradio URL in your browser.

---

## Useful `.env` settings

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5-coder:7b
MODEL_TEMPERATURE=0.2
MAX_TOOL_STEPS=5
MAX_FILE_CHARS=12000
ALLOW_CODE_EXECUTION=false
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

### Important
`ALLOW_CODE_EXECUTION` is off by default.

If you want the agent to run short Python snippets in the workspace, change:

```env
ALLOW_CODE_EXECUTION=true
```

Only do that when you are comfortable letting the agent execute code.

---

## How memory works

Memory is stored in:

```text
memory/agent_memory.db
```

There are two kinds of memory:

- **chat history** for the current session
- **durable notes** that the agent can save and search later

Examples:

- “Remember that my project uses Rust and Axum.”
- “Search my saved notes for Ollama settings.”

---

## How workspace tools work

The agent only reads and writes inside:

```text
workspace/
```

That is intentional, so it cannot wander around your whole disk.

Put your project files there if you want the agent to inspect them.

---

## Good first tests

Try messages like:

- `List the files in the workspace`
- `Save a note that my default model is qwen2.5-coder:7b`
- `Search my notes for default model`
- `Search Hugging Face for sentence embedding models`
- `Use Wikipedia to summarize LangGraph`
- `Write a file named hello.txt in the workspace with a short welcome message`

If you enable code execution:

- `Run Python code that prints the numbers 1 to 5`



## Troubleshooting

### The model does not answer
Make sure Ollama is running and the model exists:

```bash
ollama list
```

### The UI opens but responses fail
Check:

- `.env` model name
- Ollama is running
- the model is already pulled

### The agent says code execution is disabled
Set this in `.env`:

```env
ALLOW_CODE_EXECUTION=true
```

Then restart the app.

### Gradio install issues
Update pip first:

```bash
python -m pip install --upgrade pip
```

Then reinstall:

```bash
pip install -r requirements.txt --upgrade
```
