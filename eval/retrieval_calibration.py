"""Calibrate chunking, retrieval depth, fusion weight, and abstention."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from eval.validate_cases import load_cases, validate_cases
from src.rag import build_chunks, build_indexes, evaluate, load_documents

DEFAULT_CHUNK_CONFIGS = (
    (1000, 100, 300, 30),
    (1500, 75, 400, 20),
    (1500, 150, 400, 50),
    (1500, 300, 400, 100),
    (2000, 200, 500, 50),
)


def parse_chunk_config(value: str) -> tuple[int, int, int, int]:
    try:
        config = tuple(int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "chunk config must contain integers: parent_size:parent_overlap:child_size:child_overlap"
        ) from error
    if len(config) != 4:
        raise argparse.ArgumentTypeError(
            "chunk config must be parent_size:parent_overlap:child_size:child_overlap"
        )
    parent_size, parent_overlap, child_size, child_overlap = config
    if min(config) < 0 or min(parent_size, child_size) == 0:
        raise argparse.ArgumentTypeError(
            "chunk sizes must be positive and overlaps non-negative"
        )
    if child_size > parent_size:
        raise argparse.ArgumentTypeError("child size must not exceed parent size")
    if parent_overlap >= parent_size or child_overlap >= child_size:
        raise argparse.ArgumentTypeError("chunk overlap must be smaller than chunk size")
    return config


def chunk_config_dict(config: tuple[int, int, int, int]) -> dict[str, int]:
    return dict(
        zip(
            ("parent_size", "parent_overlap", "child_size", "child_overlap"),
            config,
            strict=True,
        )
    )


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
    parser.add_argument(
        "--chunk-configs",
        nargs="+",
        type=parse_chunk_config,
        default=DEFAULT_CHUNK_CONFIGS,
        metavar="PARENT:OVERLAP:CHILD:OVERLAP",
    )
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

    from langchain_huggingface import HuggingFaceEmbeddings

    pages = load_documents(args.corpus_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )

    hybrid_results = []
    best_hybrid = None
    best_assets = None
    for chunk_config in args.chunk_configs:
        chunk_config_data = chunk_config_dict(chunk_config)
        print(f"Indexing {chunk_config_data}", flush=True)
        parents, children = build_chunks(pages, *chunk_config)
        parents_by_id = {
            document.metadata["chunk_id"]: document for document in parents
        }
        bm25, vectorstore, _ = build_indexes(
            parents,
            children,
            args.device,
            max(args.candidate_k),
            ("hybrid",),
            embeddings=embeddings,
        )
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
                **chunk_config_data,
                "candidate_k": candidate_k,
                "bm25_weight": weight,
                "metrics": metrics,
            }
            hybrid_results.append(result)
            metric_names = ("recall_at_5", "mrr", "source_hit_rate")
            rank = tuple(metrics[name] for name in metric_names)
            best_rank = (
                tuple(best_hybrid["metrics"][name] for name in metric_names)
                if best_hybrid
                else None
            )
            if best_rank is None or rank > best_rank:
                best_hybrid = result
                best_assets = bm25, vectorstore, parents_by_id, predictions

    assert best_hybrid is not None and best_assets is not None
    bm25, vectorstore, parents_by_id, best_hybrid_predictions = best_assets
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("BAAI/bge-reranker-base", device=args.device)
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
            "k": args.k,
            "scope_guard": True,
            "hyde": False,
            "web_fallback": False,
        },
        "grid": {
            "chunk_configs": [
                chunk_config_dict(config) for config in args.chunk_configs
            ],
            "candidate_k": args.candidate_k,
            "bm25_weights": args.bm25_weights,
            "thresholds": args.thresholds,
        },
        "hybrid_results": hybrid_results,
        "best_hybrid": best_hybrid,
        "threshold_results": threshold_results,
        "best_reranked": {
            **{
                key: best_hybrid[key]
                for key in (
                    "parent_size",
                    "parent_overlap",
                    "child_size",
                    "child_overlap",
                    "candidate_k",
                    "bm25_weight",
                )
            },
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
        for prediction in best_hybrid_predictions + reranked_predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
