import unittest
from pathlib import Path

from eval.run_baseline import calculate_metrics
from eval.validate_cases import load_cases, validate_cases


class EvaluationCasesTest(unittest.TestCase):
    def test_versioned_cases_are_valid_and_reviewed(self) -> None:
        cases = load_cases(Path("data/eval_cases.jsonl"))

        self.assertEqual(validate_cases(cases), [])
        self.assertEqual(validate_cases(cases, require_reviewed=True), [])

    def test_baseline_metrics(self) -> None:
        predictions = [
            {
                "answerable": True,
                "regulation": ["PP Nomor 35 Tahun 2021"],
                "page": [17],
                "retrieved": [
                    {"regulation": "PP Nomor 35 Tahun 2021", "page": 17},
                ],
                "citations": [
                    {"regulation": "PP Nomor 35 Tahun 2021", "page": 17},
                ],
                "status": "answer",
                "latency_seconds": 2.0,
                "gpu_peak_memory_mb": 100,
            },
            {
                "answerable": False,
                "regulation": [],
                "page": [],
                "retrieved": [],
                "citations": [],
                "status": "insufficient_context",
                "latency_seconds": 1.0,
                "gpu_peak_memory_mb": 80,
            },
        ]

        metrics = calculate_metrics(predictions)

        self.assertEqual(metrics["retrieval"]["recall_at_5"], 1)
        self.assertEqual(metrics["retrieval"]["mrr"], 1)
        self.assertEqual(metrics["generation"]["citation_precision"], 1)
        self.assertEqual(metrics["safety"]["abstention_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
