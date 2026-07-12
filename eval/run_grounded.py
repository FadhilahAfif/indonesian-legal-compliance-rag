"""Run the grounded-answer benchmark with the selected retrieval configuration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from eval.run_baseline import load_cases, matches_reference
from src.rag import (
    build_chunks,
    build_indexes,
    document_reference,
    generate_grounded_answer,
    insufficient_context_answer,
    load_documents,
    retrieve,
)


def citation_is_grounded(
    citation: dict[str, Any], retrieved: list[dict[str, Any]]
) -> bool:
    quote = " ".join(citation["quote"].split())
    return any(
        citation["chunk_id"] == reference["chunk_id"]
        and quote in " ".join(reference["quote"].split())
        for reference in retrieved
    )


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in predictions if item["answerable"]]
    reciprocal_ranks = []
    source_hits = 0
    for item in answerable:
        ranks = [
            rank
            for rank, reference in enumerate(item["retrieved"][:5], 1)
            if matches_reference(reference, item)
        ]
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
        source_hits += any(
            matches_reference(reference, item, source_only=True)
            for reference in item["retrieved"][:5]
        )

    citations = [
        citation for item in predictions for citation in item.get("citations", [])
    ]
    generation_attempts = [
        item for item in predictions if item["retrieval_status"] == "answer"
    ]
    latencies = [item["latency_seconds"] for item in predictions]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    valid_statuses = {"answer", "insufficient_context"}
    return {
        "retrieval": {
            "recall_at_5": sum(bool(rank) for rank in reciprocal_ranks)
            / len(answerable),
            "mrr": statistics.fmean(reciprocal_ranks),
            "source_hit_rate": source_hits / len(answerable),
        },
        "generation": {
            "faithfulness": None,
            "answer_relevance": None,
            "citation_precision": sum(
                citation_is_grounded(citation, item["retrieved"])
                for item in predictions
                for citation in item.get("citations", [])
            )
            / len(citations)
            if citations
            else None,
            "attempt_count": len(generation_attempts),
            "valid_output_rate": sum(
                item["status"] in valid_statuses for item in generation_attempts
            )
            / len(generation_attempts)
            if generation_attempts
            else None,
            "note": "Faithfulness and answer relevance require manual review.",
        },
        "safety": {
            "abstention_accuracy": sum(
                item["status"] in valid_statuses
                and (item["status"] == "insufficient_context")
                == (not item["answerable"])
                for item in predictions
            )
            / len(predictions),
            "invalid_output_count": sum(
                item["status"] == "invalid_output" for item in predictions
            ),
        },
        "runtime": {
            "mean_latency_seconds": statistics.fmean(latencies),
            "p95_latency_seconds": sorted_latencies[p95_index],
            "max_gpu_memory_mb": max(
                item["gpu_peak_memory_mb"] for item in predictions
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/eval_cases.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval/results/grounded-generation")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    cases = load_cases(args.cases, args.limit)
    pages = load_documents(args.corpus_dir)
    parents, children = build_chunks(pages, 1000, 100, 300, 30)
    parents_by_id = {document.metadata["chunk_id"]: document for document in parents}
    bm25, vectorstore, reranker = build_indexes(
        parents, children, device, 10, ("hybrid_rerank",)
    )

    predictions = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        documents, scores, retrieval_status = retrieve(
            case["question"],
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
        references = [
            document_reference(document, scores[position])
            for position, document in enumerate(documents)
        ]
        if retrieval_status == "insufficient_context":
            answer = insufficient_context_answer()
        else:
            answer = generate_grounded_answer(case["question"], documents)
        status = answer["status"]
        predictions.append(
            {
                **case,
                "retrieval_status": retrieval_status,
                "status": status,
                "answer": answer,
                "retrieved": references,
                "citations": answer.get("citations", []),
                "latency_seconds": time.perf_counter() - started,
                "gpu_peak_memory_mb": (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if device == "cuda"
                    else 0.0
                ),
            }
        )

    metrics = calculate_metrics(predictions)
    failures = [
        item["id"]
        for item in predictions
        if item["status"] not in {"answer", "insufficient_context"}
        or (item["status"] == "insufficient_context") != (not item["answerable"])
        or (
            item["answerable"]
            and not any(
                matches_reference(reference, item) for reference in item["retrieved"]
            )
        )
    ]
    report = {
        "config": {
            "generator": "deterministic_extractive",
            "device": device,
            "seed": 42,
            "parent_chunk_size": 1000,
            "parent_chunk_overlap": 100,
            "child_chunk_size": 300,
            "child_chunk_overlap": 30,
            "candidate_k": 10,
            "rerank_k": 5,
            "generation_context_k": 3,
            "bm25_weight": 0.4,
            "reranker_threshold": 0.3,
            "deterministic_generation": True,
            "hyde": False,
            "web_fallback": False,
        },
        "requested_case_count": len(cases),
        "case_count": len(predictions),
        "metrics": metrics,
        "failure_cases": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "grounded_report.json"
    predictions_path = args.output_dir / "grounded_predictions.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {report_path} and {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
