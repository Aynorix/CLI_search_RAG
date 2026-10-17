# Assignment 3: PDF RAG (Retrieval-Augmented Generation) System

A command-line tool that transforms a document search tool into a full RAG system. Instead of merely printing matching chunks and similarity scores, the system retrieves relevant document chunks, constructs an augmented prompt, and queries an LLM to generate a grounded, factual response.

---

## Example Output

```text
PS C:\Users\Aynorix\Downloads\CLI_search_tool-main\CLI_search_tool-main> .\.venv\Scripts\python.exe search_pdf.py "cloud computing technology"

PDF: C:\Users\Aynorix\Downloads\CLI_search_tool-main\CLI_search_tool-main\data\assignment1-pdf.pdf
Indexed 100 of 100 pages with selectable text.
Query: cloud computing technology
Retrieved 3 relevant chunk(s) from document.

================================================================================
AUGMENTED PROMPT:
================================================================================
You are a helpful and factual assistant. Use the following retrieved context chunks from the PDF document to answer the user's question accurately.
If the answer cannot be determined from the provided context, state clearly: "Based on the provided document chunks, I cannot find sufficient information to answer this question."
Do not speculate or include information that is not supported by the context chunks.

Retrieved Context Chunks:
================================================================================
[Chunk 1 | Page 2]:
Cloud Computing Technology

[Chunk 2 | Page 3]:
Huawei Technologies Co., Ltd. Cloud Computing Technology

[Chunk 3 | Page 1]:
CLOUD COMPUTING TECHNOLOGY Huawei Technologies Co., Ltd. Official Textbooks for Huawei ICT Academy
================================================================================

User Question:
cloud computing technology

Answer:
================================================================================

================================================================================
GENERATED RESPONSE (OPENAI - gpt-4o-mini):
================================================================================
Based on the provided context chunks from the official Huawei ICT Academy textbook,
"Cloud Computing Technology" is an instructional subject and textbook published by
Huawei Technologies Co., Ltd. that introduces foundational principles, architecture,
and implementations of cloud computing.
================================================================================
```

---

## How it Works (RAG Pipeline)

1. **Retrieval**:
   The PDF is parsed using `pypdf`, indexing selectable text across all pages. For a given query, the tool uses TF-IDF cosine similarity to rank and extract the top-$k$ most relevant text chunks along with page metadata.

2. **Augmentation**:
   The retrieved context chunks and the user query are assembled into an augmented prompt:
   - System instructions enforce factual grounding and prevent hallucinations.
   - Retrieved chunks are presented with rank and source page numbers (`[Chunk X | Page Y]`).
   - The user query is framed as the target question.

3. **Generation**:
   The augmented prompt is passed to an LLM to generate an answer grounded exclusively in the retrieved context.

4. **Output**:
   Outputs both the **Augmented Prompt** and the **Generated Response**.

---

## Setup

1. Open PowerShell in this directory.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## LLM Configuration

The tool supports multiple LLM providers. Copy `.env.example` to `.env` and configure your preferred provider:

```powershell
Copy-Item .env.example .env
```

### Supported Providers:

- **OpenRouter (Default Configured)**:
  Set `OPENROUTER_API_KEY=sk-or-v1-...` in `.env` (uses `openai/gpt-4o-mini` by default).
- **OpenAI**:
  Set `OPENAI_API_KEY=sk-...` in `.env` (uses `gpt-4o-mini` by default).
- **Groq (Free & Ultra-Fast)**:
  Set `GROQ_API_KEY=gsk_...` in `.env` (uses `llama-3.3-70b-versatile` by default).
- **Google Gemini**:
  Set `GEMINI_API_KEY=AIza...` in `.env` (uses `gemini-1.5-flash` by default).
- **Local Ollama**:
  Set `OPENAI_BASE_URL=http://localhost:11434/v1` and `OPENAI_API_KEY=ollama`.
- **Offline / Demonstration Mode**:
  Run with `--mock` or press Enter when prompted for an API key to test the full pipeline and augmented prompt generation offline without API credits.

---

## Usage

### 1. Interactive Mode
Run without arguments to be prompted for your query and settings:
```powershell
.\.venv\Scripts\python.exe search_pdf.py
```

### 2. Command-Line Arguments
Provide your question directly:
```powershell
.\.venv\Scripts\python.exe search_pdf.py "cloud computing technology"
```

Specify the number of context chunks to retrieve:
```powershell
.\.venv\Scripts\python.exe search_pdf.py --top-k 3 "basic support of cloud computing"
```

Use a specific model or provider:
```powershell
.\.venv\Scripts\python.exe search_pdf.py --provider groq --model llama-3.3-70b-versatile "what is virtualization"
```

Run in offline demonstration mode:
```powershell
.\.venv\Scripts\python.exe search_pdf.py --mock "cloud computing technology"
```

Optionally inspect raw retrieved chunks and similarity scores:
```powershell
.\.venv\Scripts\python.exe search_pdf.py --show-chunks --mock "cloud computing technology"
```
