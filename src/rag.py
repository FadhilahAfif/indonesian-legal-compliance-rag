"""Deterministic retrieval ablations for the historical legal corpus."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable

REGULATION_BY_FILE = {
    "PP Nomor 5 Tahun 2021.pdf": "PP Nomor 5 Tahun 2021",
    "PP Nomor 35 Tahun 2021.pdf": "PP Nomor 35 Tahun 2021",
    "PP Nomor 51 Tahun 2023.pdf": "PP Nomor 51 Tahun 2023",
    "UU Nomor 6 Tahun 2023.pdf": "UU Nomor 6 Tahun 2023",
}
TOPIC_BY_REGULATION = {
    "PP Nomor 5 Tahun 2021": "Perizinan berusaha berbasis risiko",
    "PP Nomor 35 Tahun 2021": "Ketenagakerjaan",
    "PP Nomor 51 Tahun 2023": "Pengupahan",
    "UU Nomor 6 Tahun 2023": "Cipta kerja",
}
REQUIRED_METADATA = {"regulation", "source", "page", "article", "topic", "chunk_id"}
ARTICLE_PATTERN = re.compile(r"\bPasal\s+(\d+(?:\s+\d+)?[A-Z]?)\b", re.IGNORECASE)
REGULATION_PATTERN = re.compile(
    r"\b(PP|UU)\s+(?:(?:Nomor|No\.?)\s+)?(\d+)\s+Tahun\s+(\d{4})\b",
    re.IGNORECASE,
)
DISCLAIMER = (
    "Informasi ini bukan nasihat hukum dan perlu diverifikasi terhadap regulasi resmi "
    "atau penasihat hukum yang berkualifikasi."
)
INSUFFICIENT_MESSAGE = (
    "Konteks yang tersedia tidak cukup untuk menjawab pertanyaan ini secara andal."
)
MAX_GENERATION_SOURCES = 3
SNIPPET_WIDTH = 220
QUESTION_STOPWORDS = {
    "apa",
    "apakah",
    "atau",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "pada",
    "secara",
    "untuk",
    "yang",
}


def extract_articles(text: str) -> str | None:
    articles = []
    for number in ARTICLE_PATTERN.findall(text):
        article = f"Pasal {' '.join(number.split())}"
        if article not in articles:
            articles.append(article)
    return ", ".join(articles) or None


def normalize_documents(documents: Iterable[Any]) -> list[Any]:
    from langchain_core.documents import Document

    normalized = []
    for document in documents:
        metadata = dict(document.metadata)
        source = re.split(r"[\\/]", str(metadata["source"]))[-1]
        regulation = metadata.get("regulation") or REGULATION_BY_FILE[source]
        page = int(metadata["page"]) + 1
        metadata.update(
            regulation=regulation,
            source=source,
            page=page,
            article=extract_articles(document.page_content),
            topic=TOPIC_BY_REGULATION[regulation],
            chunk_id=f"{regulation.lower().replace(' ', '-')}-p{page:04d}",
        )
        normalized.append(Document(page_content=document.page_content, metadata=metadata))
    return normalized


def load_documents(corpus_dir: Path) -> list[Any]:
    from langchain_community.document_loaders import PyPDFLoader

    missing = [name for name in REGULATION_BY_FILE if not (corpus_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing corpus files: {', '.join(missing)}")

    documents = []
    for filename, regulation in REGULATION_BY_FILE.items():
        pages = PyPDFLoader(str(corpus_dir / filename)).load()
        for page in pages:
            page.metadata.update(source=filename, regulation=regulation)
        documents.extend(normalize_documents(pages))
        print(f"Loaded {filename}: {len(pages)} pages", flush=True)
    return documents


def build_chunks(
    pages: list[Any],
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[Any], list[Any]]:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        add_start_index=True,
    )
    parents = []
    children = []
    for page in pages:
        for parent_number, parent in enumerate(parent_splitter.split_documents([page]), 1):
            parent_id = f"{page.metadata['chunk_id']}-r{parent_number:03d}"
            parent_metadata = {
                **parent.metadata,
                "article": extract_articles(parent.page_content) or page.metadata["article"],
                "chunk_id": parent_id,
            }
            parent = Document(page_content=parent.page_content, metadata=parent_metadata)
            parents.append(parent)
            for child_number, child in enumerate(child_splitter.split_documents([parent]), 1):
                child.metadata.update(
                    parent_id=parent_id,
                    chunk_id=f"{parent_id}-c{child_number:03d}",
                )
                children.append(child)
    return parents, children


def document_key(document: Any) -> str:
    return str(document.metadata.get("chunk_id") or document.page_content)


def deduplicate_documents(documents: Iterable[Any]) -> list[Any]:
    unique = {}
    for document in documents:
        unique.setdefault(document_key(document), document)
    return list(unique.values())


def reciprocal_rank_fusion(
    rankings: list[list[Any]], weights: list[float], constant: int = 60
) -> list[Any]:
    scores: dict[str, float] = {}
    documents: dict[str, Any] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, document in enumerate(deduplicate_documents(ranking), 1):
            key = document_key(document)
            documents[key] = document
            scores[key] = scores.get(key, 0) + weight / (constant + rank)
    return sorted(documents.values(), key=lambda document: scores[document_key(document)], reverse=True)


def dense_parent_results(
    query: str, vectorstore: Any, parents_by_id: dict[str, Any], k: int
) -> list[Any]:
    children = vectorstore.similarity_search(query, k=k)
    return deduplicate_documents(
        parents_by_id[child.metadata["parent_id"]] for child in children
    )


def build_indexes(
    parents: list[Any],
    children: list[Any],
    device: str,
    candidate_k: int,
    methods: Iterable[str],
    embeddings: Any = None,
) -> tuple[Any, Any, Any]:
    methods = tuple(methods)
    needs_sparse = any(method != "dense" for method in methods)
    needs_dense = any(method != "bm25" for method in methods)

    bm25 = None
    if needs_sparse:
        from langchain_community.retrievers import BM25Retriever

        bm25 = BM25Retriever.from_documents(parents, k=candidate_k)

    vectorstore = None
    if needs_dense:
        from langchain_community.vectorstores import FAISS

        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        # Child chunks enter the vector index exactly once.
        vectorstore = FAISS.from_documents(children, embeddings)

    reranker = None
    if "hybrid_rerank" in methods:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
    return bm25, vectorstore, reranker


def retrieve(
    query: str,
    method: str,
    k: int,
    bm25: Any = None,
    vectorstore: Any = None,
    parents_by_id: dict[str, Any] | None = None,
    reranker: Any = None,
    threshold: float = 0.1,
    bm25_weight: float = 0.4,
    candidate_k: int | None = None,
) -> tuple[list[Any], list[float], str]:
    mentioned_regulations = {
        f"{kind.upper()} Nomor {number} Tahun {year}"
        for kind, number, year in REGULATION_PATTERN.findall(query)
    }
    if not mentioned_regulations <= TOPIC_BY_REGULATION.keys():
        return [], [], "insufficient_context"

    candidate_k = candidate_k if candidate_k is not None else max(k * 2, 10)
    sparse = bm25.invoke(query)[:candidate_k] if bm25 is not None else []
    dense = (
        dense_parent_results(query, vectorstore, parents_by_id or {}, candidate_k)
        if vectorstore is not None
        else []
    )
    if method == "bm25":
        ranked = sparse
    elif method == "dense":
        ranked = dense
    else:
        ranked = reciprocal_rank_fusion(
            [sparse, dense], [bm25_weight, 1 - bm25_weight]
        )

    ranked = deduplicate_documents(ranked)
    scores: list[float] = []
    if method.endswith("_rerank") and ranked:
        scores = [
            float(score)
            for score in reranker.predict(
                [[query, document.page_content] for document in ranked[:candidate_k]]
            )
        ]
        ranked = [
            document
            for _, document in sorted(
                zip(scores, ranked[:candidate_k], strict=True),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]
        scores.sort(reverse=True)
    status = (
        "insufficient_context"
        if not ranked or (scores and scores[0] < threshold)
        else "answer"
    )
    return ranked[:k], scores[:k], status


def document_reference(document: Any, score: float | None = None) -> dict[str, Any]:
    reference = {
        key: document.metadata[key]
        for key in ("regulation", "source", "page", "article", "topic", "chunk_id")
    }
    reference["quote"] = document.page_content
    if score is not None:
        reference["score"] = score
    return reference


def _evidence_snippets(documents: list[Any]) -> dict[str, dict[str, Any]]:
    snippets = {}
    for source_index, document in enumerate(
        documents[:MAX_GENERATION_SOURCES], 1
    ):
        source_id = f"S{source_index}"
        normalized = " ".join(document.page_content.split())
        for quote_index, text in enumerate(
            textwrap.wrap(normalized, width=SNIPPET_WIDTH), 1
        ):
            snippets[f"{source_id}Q{quote_index}"] = {
                "source_id": source_id,
                "document": document,
                "text": text,
            }
    return snippets


def insufficient_context_answer() -> dict[str, Any]:
    return {
        "status": "insufficient_context",
        "short_answer": {"text": INSUFFICIENT_MESSAGE, "citations": []},
        "legal_basis": [],
        "application": [],
        "practical_steps": [],
        "limitations": ["Tidak ada bukti yang cukup dalam empat regulasi historis."],
        "citations": [],
        "disclaimer": DISCLAIMER,
    }


def generate_grounded_answer(
    question: str, documents: list[Any]
) -> dict[str, Any]:
    if not documents:
        return insufficient_context_answer()

    snippets = _evidence_snippets(documents)
    if not snippets:
        return insufficient_context_answer()
    question_terms = set(re.findall(r"\w{3,}", question.casefold())) - QUESTION_STOPWORDS
    snippet_id, snippet = max(
        snippets.items(),
        key=lambda item: len(
            question_terms & set(re.findall(r"\w{3,}", item[1]["text"].casefold()))
        ),
    )
    reference = document_reference(snippet["document"])
    reference.update(id=snippet_id, quote=snippet["text"])
    article = reference["article"] or f"halaman {reference['page']}"
    citation_ids = [snippet_id]
    return {
        "status": "answer",
        "short_answer": {"text": snippet["text"], "citations": citation_ids},
        "legal_basis": [
            {
                "text": f"{reference['regulation']}, {article}: {snippet['text']}",
                "citations": citation_ids,
            }
        ],
        "application": [],
        "practical_steps": [],
        "limitations": [
            "Jawaban bersifat ekstraktif; penerapan pada fakta pengguna tidak disimpulkan otomatis."
        ],
        "citations": [reference],
        "disclaimer": DISCLAIMER,
    }


def evaluate(
    cases: list[dict[str, Any]],
    method: str,
    k: int,
    **retrieval_args: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    reciprocal_ranks = []
    source_hits = 0
    latencies = []
    for case in cases:
        started = time.perf_counter()
        documents, scores, status = retrieve(
            case["question"], method, k, **retrieval_args
        )
        latencies.append(time.perf_counter() - started)
        references = [
            document_reference(document, scores[index] if scores else None)
            for index, document in enumerate(documents)
        ]
        predictions.append(
            {
                "method": method,
                "id": case["id"],
                "answerable": case["answerable"],
                "status": status,
                "retrieved": references,
            }
        )
        if case["answerable"]:
            ranks = [
                rank
                for rank, reference in enumerate(references, 1)
                if reference["regulation"] in case["regulation"]
                and reference["page"] in case["page"]
            ]
            reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
            source_hits += any(
                reference["regulation"] in case["regulation"]
                for reference in references
            )

    answerable_count = len(reciprocal_ranks)
    metrics = {
        "recall_at_5": sum(bool(rank) for rank in reciprocal_ranks) / answerable_count,
        "mrr": statistics.fmean(reciprocal_ranks),
        "source_hit_rate": source_hits / answerable_count,
        "abstention_accuracy": sum(
            (prediction["status"] == "insufficient_context")
            == (not prediction["answerable"])
            for prediction in predictions
        )
        / len(predictions),
        "mean_latency_seconds": statistics.fmean(latencies),
        "duplicate_results": sum(
            len(item["retrieved"])
            - len({reference["chunk_id"] for reference in item["retrieved"]})
            for item in predictions
        ),
        "valid_metadata": all(
            REQUIRED_METADATA <= reference.keys()
            and isinstance(reference["page"], int)
            and reference["page"] >= 1
            for item in predictions
            for reference in item["retrieved"]
        ),
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/eval_cases.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "dense", "hybrid", "hybrid_rerank"),
        default=("bm25", "dense", "hybrid", "hybrid_rerank"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-size", type=int, default=1500)
    parser.add_argument("--parent-overlap", type=int, default=150)
    parser.add_argument("--child-size", type=int, default=400)
    parser.add_argument("--child-overlap", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.candidate_k < args.k:
        parser.error("candidate-k must be greater than or equal to k")
    if not 0 <= args.bm25_weight <= 1:
        parser.error("bm25-weight must be between 0 and 1")

    from eval.validate_cases import load_cases, validate_cases
    cases = load_cases(args.cases)
    errors = validate_cases(cases, require_reviewed=True)
    if errors:
        raise ValueError("Invalid evaluation set:\n- " + "\n- ".join(errors))

    pages = load_documents(args.corpus_dir)
    parents, children = build_chunks(
        pages,
        args.parent_size,
        args.parent_overlap,
        args.child_size,
        args.child_overlap,
    )
    parents_by_id = {document.metadata["chunk_id"]: document for document in parents}
    bm25, vectorstore, reranker = build_indexes(
        parents, children, args.device, args.candidate_k, args.methods
    )

    results = {}
    predictions = []
    for method in args.methods:
        print(f"Evaluating {method}", flush=True)
        metrics, method_predictions = evaluate(
            cases,
            method,
            args.k,
            bm25=bm25 if method != "dense" else None,
            vectorstore=vectorstore if method != "bm25" else None,
            parents_by_id=parents_by_id,
            reranker=reranker,
            threshold=args.threshold,
            bm25_weight=args.bm25_weight,
            candidate_k=args.candidate_k,
        )
        results[method] = metrics
        predictions.extend(method_predictions)

    report = {
        "case_count": len(cases),
        "config": {
            "parent_size": args.parent_size,
            "parent_overlap": args.parent_overlap,
            "child_size": args.child_size,
            "child_overlap": args.child_overlap,
            "k": args.k,
            "candidate_k": args.candidate_k,
            "bm25_weight": args.bm25_weight,
            "threshold": args.threshold,
            "scope_guard": True,
            "hyde": False,
            "web_fallback": False,
            "duplicate_child_indexing": False,
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "retrieval_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
