"""Assignment 3: RAG (Retrieval-Augmented Generation) Command Line Tool.

Retrieves relevant chunks from a PDF based on a user query, constructs an augmented prompt,
and queries an LLM to generate a factual response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
from collections import Counter
from typing import Any

# Optional: Load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pypdf import PdfReader


DEFAULT_PDF = Path(__file__).parent / "data" / "assignment1-pdf.pdf"
WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

DEFAULT_MODELS = {
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3",
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
}


@dataclass
class RetrievedChunk:
    """Represents a text chunk retrieved from the PDF document."""

    rank: int
    page_number: int
    score: float
    content: str


def extract_pages(pdf_path: Path) -> tuple[int, list[tuple[int, str]]]:
    """Return (total_pages, list of (1-based page number, selectable text))."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = unicodedata.normalize("NFKC", page.extract_text() or "")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append((page_number, text))
        return len(reader.pages), pages
    except Exception as exc:
        raise RuntimeError(f"Could not read '{pdf_path}': {exc}") from exc


def make_chunk(text: str, query: str, max_chars: int = 700) -> str:
    """Extract a coherent chunk of text around the most relevant sentences for the query."""
    if len(text) <= max_chars:
        return text

    query_terms = {word.lower() for word in WORD_RE.findall(query)}
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text[:max_chars]

    def overlap(sentence: str) -> tuple[int, int]:
        terms = {word.lower() for word in WORD_RE.findall(sentence)}
        return len(query_terms & terms), -len(sentence)

    best_idx = max(range(len(sentences)), key=lambda i: overlap(sentences[i]))

    # Expand a contiguous window around the best sentence up to max_chars
    start = best_idx
    end = best_idx + 1
    current_len = len(sentences[best_idx])

    while current_len < max_chars:
        expanded = False
        if start > 0 and current_len + len(sentences[start - 1]) + 1 <= max_chars:
            start -= 1
            current_len += len(sentences[start]) + 1
            expanded = True
        if end < len(sentences) and current_len + len(sentences[end]) + 1 <= max_chars:
            end += 1
            current_len += len(sentences[end - 1]) + 1
            expanded = True
        if not expanded:
            break

    return " ".join(sentences[start:end])


def search(
    pdf_path: Path, query: str, top_k: int
) -> tuple[int, int, list[RetrievedChunk]]:
    """Index PDF text and retrieve top_k matching chunks using TF-IDF cosine similarity."""
    total_page_count, pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(
            "No selectable text was found in this PDF. Choose a text-based PDF "
            "or OCR the scanned document first."
        )

    def terms(text: str) -> list[str]:
        words = [word.lower() for word in WORD_RE.findall(text)]
        return words + [f"{left} {right}" for left, right in zip(words, words[1:])]

    page_terms = [terms(text) for _, text in pages]
    document_frequency = Counter(term for doc in page_terms for term in set(doc))
    document_count = len(page_terms)
    idf = {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    def normalized_vector(document_terms: list[str]) -> dict[str, float]:
        frequencies = Counter(document_terms)
        weights = {
            term: (1 + math.log(count)) * idf[term]
            for term, count in frequencies.items()
            if term in idf
        }
        magnitude = math.sqrt(sum(weight * weight for weight in weights.values()))
        return (
            {term: weight / magnitude for term, weight in weights.items()}
            if magnitude
            else {}
        )

    query_vector = normalized_vector(terms(query))
    ranked = []
    for (page_number, text), document in zip(pages, page_terms):
        page_vector = normalized_vector(document)
        score = sum(
            query_weight * page_vector.get(term, 0.0)
            for term, query_weight in query_vector.items()
        )
        ranked.append((page_number, score, text))

    ranked.sort(key=lambda result: (-result[1], result[0]))

    chunks = []
    for rank, (page_num, score, text) in enumerate(ranked[: min(top_k, len(ranked))], start=1):
        if score > 0:
            chunk_content = make_chunk(text, query)
            chunks.append(
                RetrievedChunk(
                    rank=rank,
                    page_number=page_num,
                    score=score,
                    content=chunk_content,
                )
            )

    return total_page_count, len(pages), chunks


def build_augmented_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Provide the (query, chunks) formatted as a RAG augmented prompt for the LLM."""
    if not chunks:
        context_text = "No relevant context chunks were found for this query in the document."
    else:
        sections = []
        for chunk in chunks:
            sections.append(
                f"[Chunk {chunk.rank} | Page {chunk.page_number}]:\n{chunk.content}"
            )
        context_text = "\n\n".join(sections)

    prompt = f"""You are a helpful and factual assistant. Use the following retrieved context chunks from the PDF document to answer the user's question accurately.
If the answer cannot be determined from the provided context, state clearly: "Based on the provided document chunks, I cannot find sufficient information to answer this question."
Do not speculate or include information that is not supported by the context chunks.

Retrieved Context Chunks:
================================================================================
{context_text}
================================================================================

User Question:
{query}

Answer:"""
    return prompt


def resolve_llm_config(
    provider_arg: str | None = None,
    model_arg: str | None = None,
    api_key_arg: str | None = None,
    base_url_arg: str | None = None,
    mock_flag: bool = False,
) -> dict[str, Any]:
    """Detect and configure the LLM provider, API key, base URL, and model."""
    if mock_flag:
        return {"mode": "mock", "provider": "mock", "model": "mock", "api_key": None, "base_url": None}

    # API Key search
    api_key = (
        api_key_arg
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )

    # Provider identification
    provider = provider_arg
    if not provider:
        if (api_key and api_key.startswith("sk-or-v1-")) or os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.getenv("GROQ_API_KEY") or (api_key and api_key.startswith("gsk_")):
            provider = "groq"
        elif os.getenv("GEMINI_API_KEY") or (api_key and api_key.startswith("AIza")):
            provider = "gemini"
        elif os.getenv("OPENAI_BASE_URL"):
            provider = "custom"
        elif os.getenv("OPENAI_API_KEY") or (api_key and api_key.startswith("sk-")):
            provider = "openai"
        elif api_key:
            provider = "openai"
        else:
            provider = "none"

    base_url = base_url_arg or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URLS.get(provider)
    model = (
        model_arg
        or os.getenv(f"{provider.upper()}_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    )

    if provider == "ollama":
        api_key = api_key or "ollama"

    return {
        "mode": "live" if api_key or provider == "ollama" else "mock",
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


def generate_mock_response(query: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a high-quality simulated response when running offline or without an API key."""
    if not chunks:
        return (
            "Based on the provided document chunks, no relevant information was found "
            f"to answer the question '{query}'."
        )

    # Summarize key findings from chunks
    findings = []
    for c in chunks:
        # Find meaningful phrases/sentences in the chunk
        sentences = [s.strip() for s in SENTENCE_RE.split(c.content) if len(s.strip()) > 15]
        excerpt = sentences[0] if sentences else c.content.strip()[:120]
        if len(excerpt) > 160:
            excerpt = excerpt[:157] + "..."
        findings.append(f"- [Page {c.page_number}]: \"{excerpt}\"")

    synthesis = "\n".join(findings)
    pages_str = ", ".join(str(c.page_number) for c in chunks)
    return (
        f"[DEMONSTRATION / OFFLINE MODE]\n"
        f"Based on the retrieved context chunks from the document for '{query}':\n\n"
        f"{synthesis}\n\n"
        f"Summary: The document discusses topics related to '{query}' across Pages {pages_str}.\n\n"
        f"*(Note: To generate responses with a live LLM, configure OPENAI_API_KEY, GROQ_API_KEY, or "
        f"GEMINI_API_KEY in a .env file or pass --api-key.)*"
    )


def generate_llm_response(
    query: str,
    chunks: list[RetrievedChunk],
    config: dict[str, Any],
) -> tuple[str, str]:
    """Build augmented prompt and query the LLM, returning (augmented_prompt, response)."""
    augmented_prompt = build_augmented_prompt(query, chunks)

    if config.get("mode") == "mock":
        response = generate_mock_response(query, chunks)
        return augmented_prompt, response

    try:
        from openai import OpenAI

        extra_headers = {}
        if config.get("provider") == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://github.com",
                "X-Title": "CLI RAG Tool",
            }

        client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            default_headers=extra_headers if extra_headers else None,
        )
        completion = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and precise assistant that answers questions "
                        "strictly based on the retrieved context chunks."
                    ),
                },
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=0.2,
        )
        response_text = completion.choices[0].message.content or ""
        return augmented_prompt, response_text.strip()
    except Exception as exc:
        error_msg = f"[LLM Generation Error: {exc}]\nFalling back to simulated response:\n\n"
        fallback = generate_mock_response(query, chunks)
        return augmented_prompt, error_msg + fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 3: Convert the PDF search tool into a RAG (Retrieval-Augmented Generation) system."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question or query to search for (omit to enter interactively)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of matching context chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Path to the PDF file (default: data/assignment1-pdf.pdf)",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai", "groq", "gemini", "ollama"],
        default=None,
        help="LLM provider (default: auto-detected from environment)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., openai/gpt-4o-mini, llama-3.3-70b-versatile, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the chosen LLM provider",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for OpenAI-compatible LLM endpoints",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline demonstration mode without invoking live LLM APIs",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also display raw retrieved chunks with similarity scores before the prompt",
    )
    return parser.parse_args()


def ask_for_query() -> str:
    return input("What should I search for? ").strip()


def ask_for_top_k(default: int = 5) -> int:
    while True:
        value = input(f"How many matching chunks should I retrieve? [{default}]: ").strip()
        if not value:
            return default
        try:
            top_k = int(value)
        except ValueError:
            print("Enter a whole number, such as 5.")
            continue
        if top_k > 0:
            return top_k
        print("Enter a number greater than zero.")


def ask_for_api_key_if_needed(config: dict[str, Any], interactive: bool) -> dict[str, Any]:
    """Prompt user interactively for an API key if none is configured."""
    if config["mode"] == "live" or not interactive:
        return config

    print("\nNo LLM API key detected (OPENROUTER_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY).")
    user_key = input(
        "Enter your API key (OpenRouter/OpenAI/Groq/Gemini), or press Enter to run in offline demo mode: "
    ).strip()

    if user_key:
        return resolve_llm_config(api_key_arg=user_key)
    return config


def main() -> int:
    args = parse_args()
    interactive = not args.query
    pdf_path = args.pdf

    try:
        query = " ".join(args.query).strip() if args.query else ask_for_query()
        top_k = (
            args.top_k
            if args.top_k is not None
            else (ask_for_top_k() if interactive else 5)
        )
    except EOFError:
        print("\nInput ended before search could start.", file=sys.stderr)
        return 2

    if not query:
        print("Error: enter a non-empty search query.", file=sys.stderr)
        return 2
    if top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2
    if not pdf_path.is_file():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    # Step 1: Retrieval (Search PDF for relevant chunks)
    try:
        total_page_count, searchable_page_count, chunks = search(pdf_path, query, top_k)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nPDF: {pdf_path}")
    print(f"Indexed {searchable_page_count} of {total_page_count} pages with selectable text.")
    print(f"Query: {query}")
    print(f"Retrieved {len(chunks)} relevant chunk(s) from document.")

    # Optional: Display raw chunks if explicitly requested with --show-chunks
    if args.show_chunks:
        print(f"\n--- Retrieved Chunks ({len(chunks)}) ---")
        for chunk in chunks:
            print(f"\nChunk {chunk.rank}. Page {chunk.page_number}  (similarity: {chunk.score:.3f})")
            print(f"   {chunk.content}")

    # Step 2: Configure LLM
    llm_config = resolve_llm_config(
        provider_arg=args.provider,
        model_arg=args.model,
        api_key_arg=args.api_key,
        base_url_arg=args.base_url,
        mock_flag=args.mock,
    )
    llm_config = ask_for_api_key_if_needed(llm_config, interactive)

    # Step 3 & 4: Provide (query, chunks) to LLM and generate response
    augmented_prompt, response = generate_llm_response(query, chunks, llm_config)

    # Step 5: Output augmented prompt along with generated response
    separator = "=" * 80
    print(f"\n{separator}")
    print("AUGMENTED PROMPT:")
    print(separator)
    print(augmented_prompt)
    print(f"{separator}")

    provider_label = (
        f"{llm_config.get('provider', 'LLM').upper()} - {llm_config.get('model')}"
        if llm_config.get("mode") == "live"
        else "OFFLINE DEMO MODE"
    )
    print(f"\n{separator}")
    print(f"GENERATED RESPONSE ({provider_label}):")
    print(separator)
    print(response)
    print(f"{separator}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
