# Paper-to-Presentation & Viva Agent

An autonomous multi-agent AI research assistant that transforms academic research papers (PDF) into structured executive summaries, technical concept breakdowns, formatted 16:9 PowerPoint presentations (`.pptx`), university-level viva defense examinations, and citation-grounded conversational Q&A — powered by LangChain (LCEL), Google Gemini, Sentence Transformers, ChromaDB, PyMuPDF, and an Apple iOS 27 / Duo-inspired Streamlit interface.

---

## ✨ Features

| Capability | Description |
|---|---|
| **📄 Paper Summarisation** | Extracts structured academic summaries: Title, Abstract Summary, Problem Statement, Proposed Solution, Key Findings, and Conclusion with source chunk attribution. |
| **🧠 Technical Concept Extraction** | Deconstructs papers across 6 academic dimensions: Key Contributions, Methodology, Datasets Used, Results & Metrics, Limitations, and Future Work. |
| **📊 Slide Deck Generation** | Creates 10–12 structured slides with concise bullet hierarchies, presenter speaker notes, and source page traceability. |
| **📥 PowerPoint (`.pptx`) Export** | Generates and downloads 16:9 widescreen `.pptx` decks in memory via `python-pptx`, including title slide styling, footers, and native speaker notes. |
| **🎓 Viva Defense Preparation** | Generates 20 oral examination questions across 5 academic categories (*Conceptual*, *Methodological*, *Result-Based*, *Critical*, *Comparison*) with difficulty ratings, category filtering, and verified page citations. |
| **💬 Interactive Grounded Q&A** | Free-form research paper chat powered by hybrid RAG (semantic + lexical reranking), suggested exploration chips, and strict hallucination controls. |
| **🔒 Multi-Session Isolation** | Full-file SHA-256 document hashing (`doc_id`) and ChromaDB metadata filtering (`{"doc_id": {"$eq": doc_id}}`) isolate concurrent sessions and prevent cross-paper leakage. |
| **🎨 Adaptive Apple-Inspired UI** | Glassmorphic Streamlit workspace (`ui/styles.py`) featuring real-time Light / Dark mode toggling, 16:9 slide preview cards, and quota-aware error reporting. |

---

## 🏗️ Architecture

```text
User (Streamlit UI: app.py + ui/styles.py + ui/sidebar.py + ui/result_views.py)
       │
       ▼
  Intent Router & Dispatcher (agents/router.py)
       │
  ┌────┴──────────────────────┬──────────────────────────┬──────────────────────┐
  ▼                           ▼                          ▼                      ▼
Summarizer              Concept Extractor          Slide Generator         Viva Agent
(agents/summarizer.py)  (agents/concept_extractor.py) (agents/slide_generator.py) (agents/viva_agent.py)
  │                           │                          │                      │
  └───────────────────────────┴──────────────────────────┴──────────────────────┘
                                             │
                                             ▼
              Agentic RAG & Context Gatekeeper (core/rag_context.py + core/rag_pipeline.py)
          [PyMuPDF (core/pdf_loader.py) → Page-Level Splitter (core/text_splitter.py)]
          [all-MiniLM-L6-v2 384d (core/embeddings.py) → ChromaDB (core/vector_store.py)]
                                             │
                                             ▼
          Quota-Aware Gemini Client (core/llm.py: REST transport, 30s timeout, HTTP 429 guard)
                                             │
                                             ▼
                 Pydantic Output Validation (schemas/*.py) & PPTX Builder (utils/pptx_builder.py)
```

---

## 🗂️ Project Structure

```text
paper-to-presentation-agent/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Pinned/constrained Python dependencies
├── .env.example                    # Environment configuration template (placeholders only)
├── .gitignore                      # Git exclusion rules (.env, .venv, chroma_store, uploads, outputs)
├── pytest.ini                      # Pytest configuration
├── README.md                       # Project documentation
│
├── agents/                         # Four specialized LCEL agents & router
│   ├── concept_extractor.py        # Agent 2: Technical concepts across 6 categories
│   ├── router.py                   # Intent classifier (detect_intent) & dispatcher
│   ├── slide_generator.py          # Agent 3: 10–12 slide deck generator
│   ├── summarizer.py               # Agent 1: Structured paper summary generator
│   └── viva_agent.py               # Agent 4: 20 viva defense Q&A pairs with page attribution
│
├── config/
│   └── settings.py                 # Centralised configuration & dynamic .env loading
│
├── core/                           # Ingestion, Vector Store, LLM, and RAG modules
│   ├── embeddings.py               # Local Sentence Transformers (all-MiniLM-L6-v2) singleton
│   ├── llm.py                      # Quota-aware Gemini client (REST transport, 30s timeout)
│   ├── pdf_loader.py               # PyMuPDF validation (magic bytes, encryption) & extraction
│   ├── pdf_pipeline.py             # End-to-end PDF validation, SHA-256 doc_id, chunking
│   ├── rag_context.py              # Explicit retrieval decisions, relevance scoring, fallback
│   ├── rag_pipeline.py             # Hybrid RAG pipeline (semantic + lexical reranking + Q&A)
│   ├── text_splitter.py            # Page-level RecursiveCharacterTextSplitter (preserves page_number)
│   └── vector_store.py             # Persistent ChromaDB wrapper with doc_id isolation
│
├── prompts/                        # System prompt templates for all four agents
├── schemas/                        # Pydantic schemas for structured output validation
├── scripts/                        # Connectivity diagnostics, demos, and verification scripts
│   ├── test_gemini_connection.py   # Sanitized Gemini API connectivity probe
│   ├── verify_failure_handling.py  # Missing/invalid API key handling verification
│   └── verify_phase6.py            # Offline PPTX export & Streamlit module check
│
├── tests/                          # Automated test suite (18 test files, 209 tests)
│   └── data/
│       └── sample_research_paper.pdf
│
├── ui/                             # Streamlit UI components
│   ├── result_views.py             # Result renderers (Summary, Concepts, Slides, Viva, Chat)
│   ├── sidebar.py                  # Theme toggle, PDF upload, active document card, reset
│   └── styles.py                   # Light/Dark glassmorphic CSS design system
│
└── utils/
    ├── doc_id.py                   # Full-content SHA-256 UUID generator
    ├── file_utils.py               # PDF magic-byte validation & safe directory helpers
    └── pptx_builder.py             # 16:9 widescreen python-pptx generator with speaker notes
```

---

## 🛠️ Installation (Windows / PowerShell)

1. **Clone the repository and navigate to the project root:**
   ```powershell
   cd path\to\paper-to-presentation-agent
   ```

2. **Create and activate a Python 3.11+ virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and replace the placeholder API key with your real Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey):

```powershell
Copy-Item .env.example .env
```

Example `.env` structure (do **not** commit your actual `.env` file):

```ini
GOOGLE_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TRANSPORT=rest
GEMINI_REQUEST_TIMEOUT=30.0
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=paper_chunks
RAG_TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LLM_MAX_RETRIES=2
```

---

## 🚀 Usage

### Launch the Streamlit Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

### Workflow
1. **Upload a Research Paper**: Drop a text-based academic PDF in the sidebar. The app validates magic bytes, checks for encryption/corruption, extracts text page-by-page, chunks content with `doc_id` metadata, and indexes embeddings locally into ChromaDB.
2. **Generate Analysis**:
   - Click **⚡ Generate Full Suite** to run all four agents sequentially with live progress tracking, **or**
   - Generate outputs individually inside each tab: **Paper Summary**, **Concepts & Methodology**, **Slide Deck & PPTX**, and **Viva Defense Prep**.
3. **Download PowerPoint Deck**: In the **Slide Deck & PPTX** tab, preview all 16:9 slides and speaker notes, then click **📥 Download Presentation (.pptx)**.
4. **Prepare for Viva**: Filter the 20 generated oral defense questions by category and inspect grounded model answers and source page tags.
5. **Ask Paper (Q&A)**: Use suggested exploration chips or ask custom questions with verified page citations.
6. **Switch Appearance**: Toggle between **Dark Mode** and **Light Mode** in the sidebar.

---

## 📸 Screenshots

> Add UI captures here before final portfolio presentation:
> - **Dashboard & Paper Upload (Light / Dark Mode)**: `docs/screenshots/dashboard.png`
> - **Paper Summary & Concepts Breakdown**: `docs/screenshots/summary_concepts.png`
> - **16:9 Slide Deck Preview & PPTX Export**: `docs/screenshots/slides_preview.png`
> - **Viva Defense Q&A & Grounded Chat**: `docs/screenshots/viva_chat.png`

---

## 🧪 Testing & Verification

### 1. Run the Complete Automated Test Suite (Offline / Mocked LLM)
```powershell
.\.venv\Scripts\pytest.exe -v
```
Expected result: **209 passed** across **18 test files** covering PDF validation/extraction, SHA-256 isolation, embeddings, ChromaDB persistence & metric scoring, RAG retrieval decisions, all 4 agents, router dispatch, Pydantic schemas, PPTX generation, quota error handling, and Streamlit UI view integration.

### 2. Verify Offline PPTX Generation & UI Imports
```powershell
.\.venv\Scripts\python.exe scripts/verify_phase6.py
```

### 3. Verify Missing / Invalid API Key Error Handling
```powershell
.\.venv\Scripts\python.exe scripts/verify_failure_handling.py
```

### 4. Test Live Gemini API Connectivity (Uses 1 API Request)
```powershell
.\.venv\Scripts\python.exe scripts/test_gemini_connection.py
```

---

## ⚠️ Limitations & Known Constraints

- **Gemini Free Tier Quota**: Google AI Studio Free Tier keys enforce strict daily request limits (e.g., 20 requests/day per model bucket, and Google's backend may alias model names when tracking quota buckets). `core/llm.py` detects HTTP 429 daily exhaustion (`DailyQuotaExceededError`) and halts retries immediately to display a retry wait-time banner rather than wasting requests.
- **Text-Based PDFs Only**: PyMuPDF extracts embedded PDF text; scanned image-only PDFs without an OCR text layer are rejected with a descriptive validation error.
- **Offline Test Determinism**: The automated `pytest` suite verifies agent orchestration, schema validation, RAG retrieval, and UI rendering using deterministic `RunnableLambda` LLM mocks (`tests/conftest.py` and test mocks) so tests run reliably without consuming live API quota. Live end-to-end agent quality depends on an active, unthrottled Gemini API key.

---

## 📄 License

MIT
