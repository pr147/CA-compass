# 🧭 CA Compass — UPSC Current Affairs Copilot

> **Kaggle AI Agents: Intensive Vibe Coding Capstone — June 2026**

CA Compass is an AI-powered UPSC study assistant. Upload any current-affairs
newspaper or PDF and get a full exam-preparation workflow:

- **Topic extraction** — top 5 UPSC-relevant topics ranked by relevance score
- **Deep-dive analysis** — constitutional links, Prelims facts, Mains angles
- **Exam practice** — related PYQs, generated MCQs, Mains question
- **Study notebook** — persistent revision tracker with dashboard and study plan

---

## Kaggle Judging Criteria Map

The three required course concepts are implemented as follows.

### 1. Multi-Agent System with Google ADK

**Course day:** Day 1  
**Implementation:** `src/agents/adk_pipeline.py`

CA Compass runs a **three-agent SequentialAgent pipeline** built with
`google-adk`:

```
orchestrator (SequentialAgent)
  ├── relevance_agent  (LlmAgent + FunctionTool)  → session.state["topics_json"]
  ├── analysis_agent   (LlmAgent + FunctionTool)  → session.state["analysis_json"]
  └── exam_coach_agent (LlmAgent + FunctionTool)  → session.state["practice_json"]
```

ADK concepts demonstrated:

| Concept | Where |
|---|---|
| `LlmAgent` | Three specialist agents, each with typed tools | 
| `SequentialAgent` | `_make_orchestrator()` in `adk_pipeline.py:L190` |
| `FunctionTool` | `analyse_chunks_tool`, `deep_dive_analysis_tool`, `generate_exam_practice_tool`, `search_pyqs_tool` |
| `InMemorySessionService` | `_session_service` singleton, `adk_pipeline.py:L68` |
| Session state (`output_key`) | Each agent writes to named key; next agent reads it |
| `Runner` | `_run_agent()` in `adk_pipeline.py:L215` |

Each agent wraps an existing specialist class
(`RelevanceAgent`, `AnalysisAgent`, `ExamCoachAgent`) as an ADK `FunctionTool`,
keeping all prompt logic and heuristic fallbacks unchanged.

---

### 2. MCP Server (Model Context Protocol)

**Course day:** Day 2  
**Implementation:** `src/mcp_server.py`

CA Compass exposes a **FastMCP server** with four tools:

| MCP Tool | Description |
|---|---|
| `search_pyqs` | Keyword + subject search over the local PYQ CSV dataset |
| `notebook_save` | Persist a study session to `study_notebook.json` |
| `notebook_load` | Retrieve saved study records |
| `notebook_stats` | Return dashboard statistics |

**Running the MCP server:**

```bash
# stdio transport (for Claude Desktop / ADK MCPToolset)
python -m src.mcp_server

# SSE transport (for web clients)
python -m src.mcp_server --sse
```

**Connecting from an ADK agent:**

```python
from google.adk.tools import MCPToolset
from mcp import StdioServerParameters

toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python", args=["-m", "src.mcp_server"]
    )
)
```

The server follows the full MCP specification: tools are discoverable via
`list_tools()` and invokable via `call_tool()` with typed parameters.

---

### 3. Security Features / Guardrails

**Course day:** Day 4  
**Implementation:** `src/security.py`

Five layers of defence applied before any content reaches an LLM:

| Layer | Implementation | Where in code |
|---|---|---|
| File validation | Extension, size (≤ 50 MB), non-empty | `validate_upload()` |
| Page count limit | Hard cap of 300 pages | `validate_page_count()` |
| UPSC relevance gate | Reject off-topic documents | `validate_upsc_relevance()` |
| Prompt injection detection | 12 regex patterns; sentence-level redaction | `sanitise_chunk()` |
| API key guard | Clear message, graceful heuristic fallback | `validate_api_key()` |

**Prompt injection defence** — PDF text is attacker-controlled content.
The `sanitise_chunks()` function scans every chunk for patterns that attempt
to override system instructions (e.g. "ignore all previous instructions",
`<system>` tags, "you are now a different AI"). Detected sentences are
replaced with `[Content redacted: policy violation]` and a warning is logged.
The sanitised chunk is still processed — only the offending sentence is removed.

The UI displays a **Security & Validation Report** after every PDF analysis,
showing file size, page count, chunks processed, and injection scan result.

---

## Architecture

```
app.py                          ← Streamlit UI (unchanged structure)
│
├── src/
│   ├── agents/
│   │   ├── adk_pipeline.py     ← ADK SequentialAgent + Runner + Sessions  ★ NEW
│   │   ├── relevance_agent.py  ← RelevanceAgent (Gemini + heuristic)
│   │   ├── analysis_agent.py   ← AnalysisAgent  (Gemini + heuristic)
│   │   └── exam_coach_agent.py ← ExamCoachAgent (Gemini + heuristic + PYQ search)
│   │
│   ├── mcp_server.py           ← FastMCP server (4 tools)                 ★ NEW
│   ├── security.py             ← Guardrails (5 layers)                    ★ NEW
│   │
│   ├── services/
│   │   └── study_notebook.py   ← Local JSON persistence + dashboard logic
│   │
│   ├── schemas.py              ← Pydantic models for all data structures
│   ├── pdf_parser.py           ← PyMuPDF text extraction
│   ├── chunker.py              ← Paragraph-aware text chunker
│   └── utils/
│       └── prompts.py          ← All LLM prompt templates
│
└── src/data/
    ├── pyqs_sample.csv         ← 30 UPSC previous year questions
    └── study_notebook.json     ← User's saved study sessions
```

---

## Setup

```bash
# 1. Clone / unzip the project
cd ca_compass

# 2. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key  (optional — heuristic mode works without it)
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# 5. Run the Streamlit app
streamlit run app.py

# 6. (Optional) Run the MCP server in a separate terminal
python -m src.mcp_server
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

---

## How It Works

```
User uploads PDF
      │
      ▼
[Security Layer]
  • File size & type check
  • Page count limit
  • UPSC relevance gate
  • Prompt injection scan → redact
      │
      ▼
[PDF Parser] → raw text
      │
      ▼
[Chunker] → list of text chunks
      │
      ▼
[ADK SequentialAgent Pipeline]
  ┌─────────────────────────────────────────────┐
  │  InMemorySessionService (session state)      │
  │                                             │
  │  relevance_agent  →  topics_json            │
  │       ↓                                     │
  │  analysis_agent   →  analysis_json          │
  │       ↓                                     │
  │  exam_coach_agent →  practice_json          │
  └─────────────────────────────────────────────┘
      │
      ▼
[Streamlit UI] — displays results
      │
      ▼
[StudyNotebook] — saves to local JSON
      │
      ▼
[Dashboard] — tracks progress over time
```

---

## Running Without an API Key

All three agents have heuristic fallback mode. Without `GEMINI_API_KEY`:
- `RelevanceAgent` uses keyword scoring over UPSC domain banks
- `AnalysisAgent` uses subject-tagged template responses
- `ExamCoachAgent` uses per-subject MCQ and Mains templates
- The ADK pipeline falls back to direct agent calls
- The MCP server still runs and exposes all tools

The app is **fully functional** in heuristic mode for demonstration purposes.

---

## Project Stats

| Item | Count |
|---|---|
| Python files | 12 |
| Lines of code | ~2,500 |
| Pydantic schemas | 6 |
| ADK agents | 3 (in 1 SequentialAgent) |
| MCP tools | 4 |
| Security layers | 5 |
| PYQ records | 30 (2016–2023) |
| Subjects covered | 10 UPSC domains |

---

## Recent Updates & Bug Fixes

- **PDF Parser Fix (`src/pdf_parser.py`):** Resolved the `ValueError: Document Closed` crash by caching the page count before closing the PyMuPDF document handle.
- **Clean Repository Structure:** Moved away from tracking legacy `.zip` archives, transitioning the repository to directly track unzipped raw source code files.
- **Enhanced `.gitignore`:** Added robust ignore rules for local virtual environments (`.venv/`), environment files, and individual database logs (`study_notebook.json`) to keep the repository clean and secure.
