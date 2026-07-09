"""Calibrate retrieval depth, fusion weight, and abstention threshold."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from eval.validate_cases import load_cases, validate_cases
from src.rag import build_chunks, build_indexes, evaluate, load_documents


def threshold_metrics(
    predictions: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    true_positives = true_negatives = false_positives = false_negatives = 0
    for prediction in predictions:
        score = (
            prediction["retrieved"][0].get("score", float("-inf"))
            if prediction["retrieved"]
            else float("-inf")
        )
        predicts_answer = score >= threshold
        if predicts_answer and prediction["answerable"]:
            true_positives += 1
        elif predicts_answer:
            false_positives += 1
        elif prediction["answerable"]:
            false_negatives += 1
        else:
            true_negatives += 1
    return {
        "threshold": threshold,
        "abstention_accuracy": (true_positives + true_negatives) / len(predictions),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/eval_cases.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-k", nargs="+", type=int, default=(10, 20, 40))
    parser.add_argument("--bm25-weights", nargs="+", type=float, default=(0.2, 0.4, 0.6))
    parser.add_argument("--thresholds", nargs="+", type=float, default=(0.1, 0.2, 0.24, 0.3))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    errors = validate_cases(cases, require_reviewed=True)
    if errors:
        raise ValueError("Invalid evaluation set:\n- " + "\n- ".join(errors))
    if min(args.candidate_k) < args.k:
        parser.error("candidate-k values must be greater than or equal to k")
    if any(not 0 <= weight <= 1 for weight in args.bm25_weights):
        parser.error("bm25-weights values must be between 0 and 1")

    pages = load_documents(args.corpus_dir)
    parents, children = build_chunks(pages, 1500, 150, 400, 50)
    parents_by_id = {document.metadata["chunk_id"]: document for document in parents}
    bm25, vectorstore, reranker = build_indexes(
        parents,
        children,
        args.device,
        max(args.candidate_k),
        ("hybrid_rerank",),
    )

    hybrid_results = []
    hybrid_predictions = {}
    for candidate_k, weight in product(args.candidate_k, args.bm25_weights):
        print(
            f"Evaluating candidate_k={candidate_k}, bm25_weight={weight}",
            flush=True,
        )
        metrics, predictions = evaluate(
            cases,
            "hybrid",
            args.k,
            bm25=bm25,
            vectorstore=vectorstore,
            parents_by_id=parents_by_id,
            bm25_weight=weight,
            candidate_k=candidate_k,
        )
        result = {
            "candidate_k": candidate_k,
            "bm25_weight": weight,
            "metrics": metrics,
        }
        hybrid_results.append(result)
        hybrid_predictions[(candidate_k, weight)] = predictions

    best_hybrid = max(
        hybrid_results,
        key=lambda result: (
            result["metrics"]["recall_at_5"],
            result["metrics"]["mrr"],
            result["metrics"]["source_hit_rate"],
        ),
    )
    best_key = (best_hybrid["candidate_k"], best_hybrid["bm25_weight"])
    reranked_metrics, reranked_predictions = evaluate(
        cases,
        "hybrid_rerank",
        args.k,
        bm25=bm25,
        vectorstore=vectorstore,
        parents_by_id=parents_by_id,
        reranker=reranker,
        threshold=min(args.thresholds),
        bm25_weight=best_hybrid["bm25_weight"],
        candidate_k=best_hybrid["candidate_k"],
    )
    threshold_results = [
        threshold_metrics(reranked_predictions, threshold)
        for threshold in args.thresholds
    ]
    best_threshold = max(
        threshold_results,
        key=lambda result: (
            result["abstention_accuracy"],
            result["true_negatives"],
            result["threshold"],
        ),
    )
    reranked_metrics["abstention_accuracy"] = best_threshold["abstention_accuracy"]
    for prediction in reranked_predictions:
        score = (
            prediction["retrieved"][0].get("score", float("-inf"))
            if prediction["retrieved"]
            else float("-inf")
        )
        prediction["status"] = (
            "answer"
            if score >= best_threshold["threshold"]
            else "insufficient_context"
        )

    report = {
        "case_count": len(cases),
        "fixed_config": {
            "parent_size": 1500,
            "parent_overlap": 150,
            "child_size": 400,
            "child_overlap": 50,
            "k": args.k,
            "hyde": False,
            "web_fallback": False,
        },
        "grid": {
            "candidate_k": args.candidate_k,
            "bm25_weights": args.bm25_weights,
            "thresholds": args.thresholds,
        },
        "hybrid_results": hybrid_results,
        "best_hybrid": best_hybrid,
        "threshold_results": threshold_results,
        "best_reranked": {
            "candidate_k": best_key[0],
            "bm25_weight": best_key[1],
            "threshold": best_threshold["threshold"],
            "metrics": reranked_metrics,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "retrieval_calibration_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for prediction in hybrid_predictions[best_key] + reranked_predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
