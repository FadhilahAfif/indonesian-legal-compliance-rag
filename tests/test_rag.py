import argparse
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from eval.retrieval_calibration import parse_chunk_config, threshold_metrics
from src.rag import (
    REQUIRED_METADATA,
    build_chunks,
    build_indexes,
    deduplicate_documents,
    normalize_documents,
    reciprocal_rank_fusion,
    retrieve,
)


class RagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = normalize_documents(
            [
                Document(
                    page_content="Pasal 10\nKegiatan usaha berisiko rendah.",
                    metadata={
                        "source": r"C:\corpus\PP Nomor 5 Tahun 2021.pdf",
                        "regulation": "PP Nomor 5 Tahun 2021",
                        "page": 0,
                    },
                )
            ]
        )

    def test_metadata_is_standardized_and_page_is_one_based(self) -> None:
        metadata = self.pages[0].metadata

        self.assertTrue(REQUIRED_METADATA <= metadata.keys())
        self.assertEqual(metadata["source"], "PP Nomor 5 Tahun 2021.pdf")
        self.assertEqual(metadata["page"], 1)
        self.assertEqual(metadata["article"], "Pasal 10")

    def test_chunk_ids_are_unique_and_children_reference_parents(self) -> None:
        parents, children = build_chunks(self.pages, 30, 5, 15, 3)
        parent_ids = {document.metadata["chunk_id"] for document in parents}
        child_ids = [document.metadata["chunk_id"] for document in children]

        self.assertEqual(len(child_ids), len(set(child_ids)))
        self.assertTrue(
            all(document.metadata["parent_id"] in parent_ids for document in children)
        )

    def test_retrieval_results_are_deduplicated(self) -> None:
        first = Document(page_content="a", metadata={"chunk_id": "a"})
        second = Document(page_content="b", metadata={"chunk_id": "b"})

        self.assertEqual(deduplicate_documents([first, first]), [first])
        self.assertEqual(
            reciprocal_rank_fusion([[first, first], [second, first]], [0.4, 0.6]),
            [first, second],
        )

    def test_threshold_metrics_count_answer_and_abstention_outcomes(self) -> None:
        predictions = [
            {
                "answerable": True,
                "retrieved": [{"score": 0.8}],
            },
            {
                "answerable": False,
                "retrieved": [{"score": 0.2}],
            },
        ]

        metrics = threshold_metrics(predictions, 0.5)

        self.assertEqual(metrics["abstention_accuracy"], 1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)

    def test_chunk_config_rejects_overlap_not_smaller_than_size(self) -> None:
        self.assertEqual(parse_chunk_config("1500:150:400:50"), (1500, 150, 400, 50))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_chunk_config("1500:1500:400:50")

    def test_dense_retrieval_uses_configured_candidate_depth(self) -> None:
        parent = Document(page_content="parent", metadata={"chunk_id": "parent"})
        child = Document(page_content="child", metadata={"parent_id": "parent"})

        class VectorStore:
            def similarity_search(self, query: str, k: int) -> list[Document]:
                self.k = k
                return [child]

        vectorstore = VectorStore()
        documents, _, _ = retrieve(
            "query",
            "dense",
            1,
            vectorstore=vectorstore,
            parents_by_id={"parent": parent},
            candidate_k=7,
        )

        self.assertEqual(vectorstore.k, 7)
        self.assertEqual(documents, [parent])

    def test_build_indexes_reuses_provided_embeddings(self) -> None:
        parent = Document(page_content="parent", metadata={"chunk_id": "parent"})
        child = Document(page_content="child", metadata={"parent_id": "parent"})

        _, vectorstore, _ = build_indexes(
            [parent],
            [child],
            "cpu",
            1,
            ("dense",),
            embeddings=DeterministicFakeEmbedding(size=4),
        )

        self.assertEqual(vectorstore.index.ntotal, 1)


if __name__ == "__main__":
    unittest.main()
