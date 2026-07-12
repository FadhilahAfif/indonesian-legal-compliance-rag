"""Gradio demo for the deterministic Indonesian legal RAG pipeline."""

from __future__ import annotations

import argparse
import html
import time
from pathlib import Path
from typing import Any, Callable

from src.rag import (
    build_chunks,
    build_indexes,
    document_reference,
    generate_grounded_answer,
    insufficient_context_answer,
    load_documents,
    retrieve,
)

Search = Callable[[str], tuple[list[Any], list[float], str]]
NO_SOURCES = "_Tidak ada sumber karena sistem tidak menghasilkan klaim hukum._"
ERROR_MESSAGE = (
    "Terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."
)
MARKDOWN_SPECIALS = "\\`*_{}[]()#+!|"


def escape_markdown(value: Any) -> str:
    text = html.escape(str(value), quote=False)
    return "".join(
        f"\\{char}" if char in MARKDOWN_SPECIALS else char for char in text
    )


def build_search(corpus_dir: Path, device: str) -> Search:
    pages = load_documents(corpus_dir)
    parents, children = build_chunks(pages, 1000, 100, 300, 30)
    parents_by_id = {document.metadata["chunk_id"]: document for document in parents}
    bm25, vectorstore, reranker = build_indexes(
        parents, children, device, 10, ("hybrid_rerank",)
    )

    def search(question: str) -> tuple[list[Any], list[float], str]:
        return retrieve(
            question,
            "hybrid_rerank",
            5,
            bm25=bm25,
            vectorstore=vectorstore,
            parents_by_id=parents_by_id,
            reranker=reranker,
            threshold=0.3,
            bm25_weight=0.4,
            candidate_k=10,
        )

    return search


def render_answer(answer: dict[str, Any]) -> str:
    sections = [escape_markdown(answer["short_answer"]["text"])]
    for title, key in (
        ("Dasar hukum", "legal_basis"),
        ("Penerapan", "application"),
        ("Langkah praktis", "practical_steps"),
        ("Batasan", "limitations"),
    ):
        items = answer[key]
        if items:
            lines = [
                escape_markdown(item["text"] if isinstance(item, dict) else item)
                for item in items
            ]
            sections.append(f"### {title}\n" + "\n".join(f"- {line}" for line in lines))
    sections.append(f"> {escape_markdown(answer['disclaimer'])}")
    return "\n\n".join(sections)


def render_sources(
    citations: list[dict[str, Any]], references: list[dict[str, Any]]
) -> str:
    if not citations:
        return NO_SOURCES
    scores = {reference["chunk_id"]: reference.get("score") for reference in references}
    cards = []
    for citation in citations:
        regulation = escape_markdown(citation["regulation"])
        page = escape_markdown(citation["page"])
        article = escape_markdown(citation["article"] or "Pasal tidak terdeteksi")
        score = scores.get(citation["chunk_id"])
        score_text = f" - Skor: {score:.3f}" if score is not None else ""
        quote = "\n> ".join(
            escape_markdown(line) for line in citation["quote"].splitlines()
        )
        cards.append(
            f"### {regulation}\n"
            f"**Halaman:** {page} - **Pasal:** {article}{score_text}\n\n"
            f"> {quote}"
        )
    return "\n\n---\n\n".join(cards)


def answer_question(
    question: str, history: list[dict[str, Any]] | None, search: Search
) -> tuple[list[dict[str, Any]], str, str, str, dict[str, Any]]:
    started = time.perf_counter()
    messages = list(history or []) + [
        {"role": "user", "content": escape_markdown(question)}
    ]
    try:
        documents, scores, retrieval_status = search(question)
        references = [
            document_reference(document, scores[index] if scores else None)
            for index, document in enumerate(documents)
        ]
        answer = (
            insufficient_context_answer()
            if retrieval_status == "insufficient_context"
            else generate_grounded_answer(question, documents)
        )
        messages.append({"role": "assistant", "content": render_answer(answer)})
        status = (
            "**Status:** Berhasil - bukti ditemukan."
            if answer["status"] == "answer"
            else "**Status:** Konteks tidak cukup - sistem tidak menebak."
        )
        debug = {
            "retrieval_status": retrieval_status,
            "final_status": answer["status"],
            "retrieved": references,
        }
        sources = render_sources(answer["citations"], references)
    except Exception as error:
        messages.append({"role": "assistant", "content": ERROR_MESSAGE})
        status = "**Status:** Error - jawaban tidak dibuat."
        sources = NO_SOURCES
        debug = {"error": type(error).__name__}
    elapsed = f"**Waktu respons:** {time.perf_counter() - started:.2f} detik"
    return messages, status, elapsed, sources, debug


def build_app(search: Search) -> Any:
    import gradio as gr

    def respond(question: str, history: list[dict[str, Any]] | None) -> tuple[Any, ...]:
        return "", *answer_question(question, history, search)

    with gr.Blocks(title="Asisten Kepatuhan Hukum Indonesia") as demo:
        gr.Markdown(
            "# Asisten Kepatuhan Hukum Indonesia\n"
            "Jawaban hanya menggunakan empat regulasi historis dalam corpus. "
            "Kualitas relevance benchmark saat ini **0,6167**. "
            "**Bukan pengganti penasihat hukum.**"
        )
        chatbot = gr.Chatbot(
            label="Percakapan",
            height=480,
            placeholder="Ajukan pertanyaan tentang perizinan usaha atau ketenagakerjaan.",
        )
        with gr.Row():
            question = gr.Textbox(
                label="Pertanyaan",
                placeholder="Contoh: Apa bentuk perizinan usaha berisiko menengah rendah?",
                scale=5,
            )
            send = gr.Button("Kirim", variant="primary", scale=1)
        with gr.Row():
            status = gr.Markdown("**Status:** Siap")
            elapsed = gr.Markdown("**Waktu respons:** -")
        gr.Markdown("## Sumber")
        sources = gr.Markdown(NO_SOURCES)
        gr.Examples(
            examples=[
                "Apa saja klasifikasi tingkat risiko kegiatan usaha dalam PP Nomor 5 Tahun 2021?",
                "Apa bentuk perizinan usaha berisiko menengah rendah?",
                "Berapa tarif Pajak Penghasilan badan yang berlaku tahun ini?",
            ],
            inputs=question,
            label="Contoh pertanyaan dari benchmark",
        )
        with gr.Accordion("Debug retrieval", open=False):
            debug = gr.JSON(value={}, label=None)
        clear = gr.ClearButton(
            [question, chatbot, status, elapsed, sources, debug], value="Bersihkan"
        )

        outputs = [question, chatbot, status, elapsed, sources, debug]
        send.click(respond, [question, chatbot], outputs)
        question.submit(respond, [question, chatbot], outputs)
        clear.click(
            lambda: ("**Status:** Siap", "**Waktu respons:** -", NO_SOURCES),
            outputs=[status, elapsed, sources],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    build_app(build_search(args.corpus_dir, device)).queue(
        default_concurrency_limit=1
    ).launch()


if __name__ == "__main__":
    main()
