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
    documents_from_references,
    generate_grounded_answer,
    insufficient_context_answer,
    load_documents,
    retrieve,
)


def load_retrieval_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        predictions = [json.loads(line) for line in source if line.strip()]
    by_id = {item["id"]: item for item in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("retrieval predictions contain duplicate case IDs")
    return by_id


def replay_grounded_answers(
    cases: list[dict[str, Any]], retrieval: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    missing = [case["id"] for case in cases if case["id"] not in retrieval]
    if missing:
        raise ValueError(f"retrieval artifact is missing case IDs: {', '.join(missing)}")

    predictions = []
    for case in cases:
        retrieved = retrieval[case["id"]]
        started = time.perf_counter()
        if retrieved["status"] == "insufficient_context":
            answer = insufficient_context_answer()
        else:
            answer = generate_grounded_answer(
                case["question"],
                documents_from_references(retrieved["retrieved"]),
            )
        predictions.append(
            {
                **case,
                "retrieval_status": retrieved["status"],
                "status": answer["status"],
                "answer": answer,
                "retrieved": retrieved["retrieved"],
                "citations": answer.get("citations", []),
                "latency_seconds": time.perf_counter() - started,
                "gpu_peak_memory_mb": 0.0,
            }
        )
    return predictions


def apply_review_scores(
    predictions: list[dict[str, Any]], review_path: Path
) -> None:
    data = json.loads(review_path.read_text(encoding="utf-8"))
    scores = data.get("scores")
    if not isinstance(scores, dict) or set(scores) != {
        item["id"] for item in predictions
    }:
        raise ValueError("review scores must contain every prediction ID exactly once")
    allowed = {0, 0.5, 1}
    for prediction in predictions:
        score = scores[prediction["id"]]
        if (
            not isinstance(score, dict)
            or set(score) != {"faithfulness", "answer_relevance"}
            or score["faithfulness"] not in allowed | {None}
            or score["answer_relevance"] not in allowed
        ):
            raise ValueError(f"invalid review score for {prediction['id']}")
        prediction["review"] = score


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
    faithfulness_scores = [
        item["review"]["faithfulness"]
        for item in predictions
        if item.get("review", {}).get("faithfulness") is not None
    ]
    relevance_scores = [
        item["review"]["answer_relevance"]
        for item in predictions
        if "review" in item
    ]
    return {
        "retrieval": {
            "recall_at_5": sum(bool(rank) for rank in reciprocal_ranks)
            / len(answerable),
            "mrr": statistics.fmean(reciprocal_ranks),
            "source_hit_rate": source_hits / len(answerable),
        },
        "generation": {
            "faithfulness": (
                statistics.fmean(faithfulness_scores)
                if faithfulness_scores
                else None
            ),
            "answer_relevance": (
                statistics.fmean(relevance_scores) if relevance_scores else None
            ),
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
            "reviewed_case_count": len(relevance_scores),
            "reviewed_claim_count": len(faithfulness_scores),
            "note": (
                "Faithfulness and answer relevance use the versioned manual review."
                if relevance_scores
                else "Faithfulness and answer relevance require manual review."
            ),
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
    parser.add_argument(
        "--retrieval-predictions",
        type=Path,
        help="Replay a versioned retrieval JSONL instead of rebuilding indexes.",
    )
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    cases = load_cases(args.cases, args.limit)
    if args.retrieval_predictions:
        predictions = replay_grounded_answers(
            cases, load_retrieval_predictions(args.retrieval_predictions)
        )
    else:
        pages = load_documents(args.corpus_dir)
        parents, children = build_chunks(pages, 1000, 100, 300, 30)
        parents_by_id = {
            document.metadata["chunk_id"]: document for document in parents
        }
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
            predictions.append(
                {
                    **case,
                    "retrieval_status": retrieval_status,
                    "status": answer["status"],
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

    if args.review:
        apply_review_scores(predictions, args.review)

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
            "generation_context_k": 5,
            "snippet_ranking": "bm25_word_and_character_5gram",
            "snippet_width": 180,
            "snippet_window": 2,
            "bm25_weight": 0.4,
            "reranker_threshold": 0.3,
            "deterministic_generation": True,
            "hyde": False,
            "web_fallback": False,
            "retrieval_artifact": (
                str(args.retrieval_predictions)
                if args.retrieval_predictions
                else None
            ),
            "review": str(args.review) if args.review else None,
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
