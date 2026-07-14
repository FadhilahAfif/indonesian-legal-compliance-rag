import json
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app import answer_question


class AppTest(unittest.TestCase):
    def test_demo_notebooks_are_clean_and_syntactically_valid(self) -> None:
        notebooks = {
            Path("notebooks/demo_media_capture.ipynb"): (
                "check=True",
                "app.py",
                "record_video_dir",
                "page.video.path()",
                "page.screenshot",
                "files.download",
            ),
            Path("notebooks/demo_ui.ipynb"): (
                "check=True",
                "build_search",
                "build_app",
                "share=True",
            ),
        }
        for path, markers in notebooks.items():
            with self.subTest(path=path):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)

                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertTrue(code_cells)
                for index, cell in enumerate(code_cells, 1):
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile(
                        "".join(cell["source"]),
                        f"{path}:cell-{index}",
                        "exec",
                    )

                source = "\n".join(
                    "".join(cell["source"]) for cell in code_cells
                )
                for marker in markers:
                    self.assertIn(marker, source)

    def test_chat_handles_answer_abstention_and_error(self) -> None:
        document = Document(
            page_content=(
                "Pasal 10 Kegiatan usaha berisiko rendah. "
                "[klik](http://bad.example) <script>"
            ),
            metadata={
                "regulation": "PP Nomor 5 Tahun 2021",
                "source": "PP Nomor 5 Tahun 2021.pdf",
                "page": 1,
                "article": "Pasal 10",
                "topic": "Perizinan berusaha berbasis risiko",
                "chunk_id": "pp-5-p1-r1",
            },
        )

        history, status, _, sources, debug = answer_question(
            "Apa tingkat risikonya? <script>",
            [],
            lambda _: ([document], [0.9], "answer"),
        )
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertIn("&lt;script&gt;", history[0]["content"])
        self.assertNotIn("<script>", history[0]["content"])
        self.assertIn("Pasal 10", history[-1]["content"])
        self.assertNotIn("Batasan", history[-1]["content"])
        self.assertNotIn("Jawaban bersifat ekstraktif", history[-1]["content"])
        self.assertIn("Informasi ini bukan nasihat hukum", history[-1]["content"])
        self.assertIn("PP Nomor 5 Tahun 2021", sources)
        self.assertIn("Skor: 0.900", sources)
        self.assertIn(r"\[klik\]\(http://bad.example\)", sources)
        self.assertNotIn("[klik](http://bad.example)", sources)
        self.assertNotIn("<script>", sources)
        self.assertIn("Berhasil", status)
        self.assertEqual(debug["final_status"], "answer")

        history, status, _, sources, debug = answer_question(
            "Berapa tarif pajaknya?",
            [],
            lambda _: ([], [], "insufficient_context"),
        )
        self.assertIn("tidak cukup", history[-1]["content"])
        self.assertIn("tidak cukup", status.lower())
        self.assertIn("Tidak ada sumber", sources)
        self.assertEqual(debug["final_status"], "insufficient_context")

        history, status, _, sources, debug = answer_question(
            "Apa tingkat risikonya?",
            [],
            lambda _: (_ for _ in ()).throw(RuntimeError("secret detail")),
        )
        self.assertIn("kesalahan", history[-1]["content"].lower())
        self.assertNotIn("secret detail", history[-1]["content"])
        self.assertIn("Error", status)
        self.assertIn("Tidak ada sumber", sources)
        self.assertEqual(debug, {"error": "RuntimeError"})


if __name__ == "__main__":
    unittest.main()
