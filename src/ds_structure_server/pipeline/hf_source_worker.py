from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .loader import inspect_hf_schema, stream_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--schema-sample-rows", type=int, default=32)
    parser.add_argument("--text-column")
    parser.add_argument("--id-column")
    parser.add_argument("--title-column")
    parser.add_argument("--metadata-column", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schema = inspect_hf_schema(
        args.dataset,
        split=args.split,
        config=args.config,
        sample_rows=args.schema_sample_rows,
        text_column=args.text_column,
        id_column=args.id_column,
        title_column=args.title_column,
        metadata_columns=tuple(args.metadata_column),
    )
    documents = stream_documents(
        args.dataset,
        schema.text_column,
        split=args.split,
        config=args.config,
        id_column=schema.id_column,
        title_column=schema.title_column,
        metadata_columns=schema.metadata_columns,
        limit=args.limit,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    with args.output.open("w") as handle:
        handle.write(json.dumps({"schema": schema.to_dict()}, ensure_ascii=False) + "\n")
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "doc_id": document.doc_id,
                        "text": document.text,
                        "title": document.title,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
