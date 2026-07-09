"""Run the deterministic M1 baseline on the reviewed legal benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

REGULATION_BY_FILE = {
    "PP Nomor 5 Tahun 2021.pdf": "PP Nomor 5 Tahun 2021",
    "PP Nomor 35 Tahun 2021.pdf": "PP Nomor 35 Tahun 2021",
    "PP Nomor 51 Tahun 2023.pdf": "PP Nomor 51 Tahun 2023",
    "UU Nomor 6 Tahun 2023.pdf": "UU Nomor 6 Tahun 2023",
}


def matches_reference(reference: dict[str, Any], case: dict[str, Any], source_only: bool = False) -> bool:
    if reference.get("regulation") not in case["regulation"]:
        return False
    return source_only or reference.get("page") in case["page"]


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in predictions if item["answerable"]]
    reciprocal_ranks = []
    source_hits = 0
    citation_hits = 0
    citation_count = 0

    for item in answerable:
        relevant_ranks = [
            rank
            for rank, reference in enumerate(item["retrieved"][:5], 1)
            if matches_reference(reference, item)
        ]
        reciprocal_ranks.append(1 / relevant_ranks[0] if relevant_ranks else 0)
        source_hits += any(
            matches_reference(reference, item, source_only=True)
            for reference in item["retrieved"][:5]
        )
        citation_count += len(item["citations"])
        citation_hits += sum(matches_reference(citation, item) for citation in item["citations"])

    abstention_correct = sum(
        (item["status"] == "insufficient_context") == (not item["answerable"])
        for item in predictions
    )
    latencies = [item["latency_seconds"] for item in predictions]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, round(0.95 * len(sorted_latencies)) - 1)
    retrieval_hits = sum(bool(value) for value in reciprocal_ranks)

    return {
        "retrieval": {
            "recall_at_5": retrieval_hits / len(answerable),
            "mrr": statistics.fmean(reciprocal_ranks),
            "source_hit_rate": source_hits / len(answerable),
        },
        "generation": {
            "faithfulness": None,
            "answer_relevance": None,
            "citation_precision": citation_hits / citation_count if citation_count else 0,
            "note": "Faithfulness and answer relevance require review of generated predictions.",
        },
        "safety": {"abstention_accuracy": abstention_correct / len(predictions)},
        "runtime": {
            "mean_latency_seconds": statistics.fmean(latencies),
            "p95_latency_seconds": sorted_latencies[p95_index],
            "max_gpu_memory_mb": max(item["gpu_peak_memory_mb"] for item in predictions),
        },
    }


def load_documents(corpus_dir: Path) -> list[Any]:
    from langchain_community.document_loaders import PyPDFLoader

    documents = []
    missing = [name for name in REGULATION_BY_FILE if not (corpus_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing corpus files: {', '.join(missing)}")

    for filename, regulation in REGULATION_BY_FILE.items():
        pages = PyPDFLoader(str(corpus_dir / filename)).load()
        for page in pages:
            page.metadata.update(source=filename, regulation=regulation)
        documents.extend(pages)
        print(f"Loaded {filename}: {len(pages)} pages", flush=True)
    return documents


def build_retrieval_pipeline(documents: list[Any], device: str) -> tuple[Any, Any]:
    from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.stores import InMemoryByteStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import CrossEncoder

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(
        model="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )

    # Baseline fidelity: M2 measures and removes this historical duplicate indexing.
    vectorstore = FAISS.from_documents(child_splitter.split_documents(documents), embeddings)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryByteStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity",
        search_kwargs={"k": 10},
    )
    parent_retriever.add_documents(documents)

    bm25 = BM25Retriever.from_documents(parent_splitter.split_documents(documents))
    bm25.k = 10
    hybrid = EnsembleRetriever(
        retrievers=[bm25, parent_retriever],
        weights=[0.4, 0.6],
    )
    reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
    return hybrid, reranker


def load_generator(model_name: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.float16,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def retrieve(question: str, hybrid: Any, reranker: Any, threshold: float) -> tuple[list[Any], list[float], str]:
    documents = hybrid.invoke(question)[:6]
    if not documents:
        return [], [], "insufficient_context"

    scores = reranker.predict([[question, document.page_content] for document in documents])
    ranked = sorted(zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True)
    top_score = float(ranked[0][0])
    status = "answer" if top_score >= threshold else "insufficient_context"
    return [document for _, document in ranked[:5]], [float(score) for score, _ in ranked[:5]], status


def document_reference(document: Any, score: float) -> dict[str, Any]:
    return {
        "regulation": document.metadata["regulation"],
        "source": document.metadata["source"],
        "page": int(document.metadata["page"]) + 1,
        "article": None,
        "score": score,
        "text": document.page_content,
    }


def generate_answer(question: str, documents: list[Any], model: Any, tokenizer: Any) -> str:
    import torch

    context = "\n\n".join(
        f"[{index}] {document.metadata['regulation']}, halaman "
        f"{int(document.metadata['page']) + 1}\n{document.page_content}"
        for index, document in enumerate(documents[:3], 1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah asisten kepatuhan hukum Indonesia. Jawab hanya dari konteks. "
                "Jika bukti tidak cukup, jawab persis: insufficient_context. "
                "Jangan tampilkan proses berpikir internal."
            ),
        },
        {
            "role": "user",
            "content": f"Konteks:\n{context}\n\nPertanyaan: {question}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1792,
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    from eval.validate_cases import load_cases as load_reviewed_cases
    from eval.validate_cases import validate_cases

    cases = load_reviewed_cases(path)
    errors = validate_cases(cases, require_reviewed=True)
    if errors:
        raise ValueError("Invalid evaluation set:\n- " + "\n- ".join(errors))
    return cases[:limit] if limit else cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/eval_cases.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--model", default="Slotherynn/legal-chatbot-qwen-grpo")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import set_seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. Run this baseline in Google Colab.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cases = load_cases(args.cases, args.limit)
    documents = load_documents(args.corpus_dir)
    hybrid, reranker = build_retrieval_pipeline(documents, "cuda")
    model = tokenizer = None
    if not args.retrieval_only:
        model, tokenizer = load_generator(args.model)

    predictions = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        retrieved, scores, status = retrieve(case["question"], hybrid, reranker, args.threshold)
        references = [
            document_reference(document, score)
            for document, score in zip(retrieved, scores, strict=True)
        ]
        answer = None
        if status == "answer" and model is not None and tokenizer is not None:
            answer = generate_answer(case["question"], retrieved, model, tokenizer)
            if answer.strip().lower() == "insufficient_context":
                status = "insufficient_context"

        predictions.append(
            {
                **case,
                "status": status,
                "answer": answer,
                "retrieved": references,
                "citations": references[:3] if status == "answer" else [],
                "latency_seconds": time.perf_counter() - started,
                "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        )

    metrics = calculate_metrics(predictions)
    failures = [
        item["id"]
        for item in predictions
        if (item["status"] == "insufficient_context") != (not item["answerable"])
        or (
            item["answerable"]
            and not any(matches_reference(reference, item) for reference in item["retrieved"][:5])
        )
    ]
    report = {
        "config": {
            "model": args.model,
            "seed": 42,
            "threshold": args.threshold,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "parent_chunk_size": 1500,
            "parent_chunk_overlap": 150,
            "child_chunk_size": 400,
            "child_chunk_overlap": 50,
            "retrieval_k": 10,
            "rerank_k": 5,
            "generation_context_k": 3,
            "hyde": False,
            "web_fallback": False,
            "duplicate_child_indexing": True,
            "deterministic_generation": True,
            "retrieval_only": args.retrieval_only,
        },
        "case_count": len(predictions),
        "metrics": metrics,
        "failure_cases": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "baseline_predictions.jsonl"
    report_path = args.output_dir / "baseline_report.json"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {predictions_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
