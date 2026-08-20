"""Push the corpus to a private dataset and the demo to a public Space.

Run after `huggingface-cli login`.

    python scripts/deploy_space.py --user <your-hf-username>

Two repositories, deliberately:

- a **private dataset** holding the index and parsed articles, because the Actes
  uniformes are not redistributed and a public Space repository would do exactly
  that;
- a **public Space** holding only code, which pulls the dataset at start-up
  using its own HF_TOKEN secret.

The Space needs that secret set once by hand, in Settings, since a token is a
credential and does not belong in a script or a git history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kora import paths

SPACE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "space"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Hugging Face username")
    parser.add_argument("--name", default="kora", help="Repository name for both repos")
    parser.add_argument("--hardware", default="zero-a10g", help="Space hardware SKU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    dataset_id = f"{args.user}/{args.name}-corpus"
    space_id = f"{args.user}/{args.name}"

    index_dir = paths.INDEX_DIR
    interim_dir = paths.INTERIM_DIR
    for directory in (index_dir, interim_dir):
        if not directory.exists():
            print(f"missing {directory}. Build the corpus and index first.", file=sys.stderr)
            return 1

    print(f"dataset (private): {dataset_id}")
    print(f"space   (public):  {space_id}   hardware={args.hardware}")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    api.create_repo(dataset_id, repo_type="dataset", private=True, exist_ok=True)
    for directory, prefix in ((index_dir, "indexes"), (interim_dir, "interim")):
        print(f"uploading {directory} -> {prefix}/")
        api.upload_folder(
            repo_id=dataset_id,
            repo_type="dataset",
            folder_path=str(directory),
            path_in_repo=prefix,
        )

    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    print(f"uploading {SPACE_DIR} -> space")
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )

    api.add_space_variable(space_id, "KORA_DATASET_REPO", dataset_id)

    print(f"\nSpace: https://huggingface.co/spaces/{space_id}")
    print(
        "\nOne manual step left, because it involves a credential:\n"
        f"  Settings -> Variables and secrets -> new secret HF_TOKEN\n"
        f"  with read access to {dataset_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
