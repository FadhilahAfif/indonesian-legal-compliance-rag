"""Fail fast when the documented project environment is incomplete."""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

RUNTIME_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "langchain-classic": "langchain_classic",
    "langchain-community": "langchain_community",
    "langchain-core": "langchain_core",
    "langchain-text-splitters": "langchain_text_splitters",
    "langchain-huggingface": "langchain_huggingface",
    "faiss-cpu": "faiss",
    "rank-bm25": "rank_bm25",
    "sentence-transformers": "sentence_transformers",
    "pypdf": "pypdf",
    "pymupdf": "fitz",
    "gradio": "gradio",
}

TRAINING_PACKAGES = {
    "unsloth": "unsloth",
    "datasets": "datasets",
    "trl": "trl",
    "peft": "peft",
    "wandb": "wandb",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="store_true",
        help="also validate dependencies from requirements-training.txt",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            f"Python 3.12 is required; found {sys.version.split()[0]}."
        )

    packages = RUNTIME_PACKAGES | (TRAINING_PACKAGES if args.training else {})
    failures = []

    for distribution, module in packages.items():
        try:
            importlib.import_module(module)
            print(f"OK {distribution}=={version(distribution)}", flush=True)
        except (ImportError, PackageNotFoundError) as error:
            failures.append(f"{distribution}: {error}")

    if failures:
        raise SystemExit("Environment check failed:\n- " + "\n- ".join(failures))

    print("Environment check passed.", flush=True)


if __name__ == "__main__":
    main()
