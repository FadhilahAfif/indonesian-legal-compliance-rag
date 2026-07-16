import json
import subprocess
import unittest
from pathlib import Path


PUBLIC_NOTEBOOKS = {
    "baseline_evaluation.ipynb",
    "demo_cuda_validation.ipynb",
    "demo_media_capture.ipynb",
    "demo_ui.ipynb",
    "grounded_generation_benchmark.ipynb",
    "grpo_training.ipynb",
    "model_comparison_benchmark.ipynb",
    "rag_pipeline_experiment.ipynb",
    "retrieval_ablation.ipynb",
    "retrieval_calibration.ipynb",
    "retrieval_validation.ipynb",
    "supervised_fine_tuning.ipynb",
}


class NotebookTest(unittest.TestCase):
    def test_only_curated_public_notebooks_are_tracked(self) -> None:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", "notebooks/*.ipynb"],
            text=True,
        ).splitlines()
        paths = [Path(path) for path in tracked]
        self.assertEqual({path.name for path in paths}, PUBLIC_NOTEBOOKS)

        readme = Path("README.md").read_text(encoding="utf-8")
        for path in paths:
            with self.subTest(path=path):
                self.assertRegex(path.name, r"^[a-z0-9_]+\.ipynb$")
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)

                code = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertNotIn("dev/apiip", code)
                if "github.com/FadhilahAfif/indonesian-legal-compliance-rag.git" in code:
                    self.assertIn('BRANCH = "main"', code)
                    self.assertIn("checkout", code)
                if "push_to_hub_merged" in code:
                    self.assertIn("OUTPUT_MODEL_ID", code)
                    self.assertNotRegex(
                        code,
                        r'push_to_hub_merged\(\s*"Slotherynn/',
                    )

                self.assertIn(f"notebooks/{path.name}", readme)
                self.assertNotIn(
                    "unggah ZIP tersebut kembali ke percakapan",
                    path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
