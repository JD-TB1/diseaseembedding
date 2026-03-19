#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EMBED_STATS_JSON,
    DEFAULT_EXPORT_TSV,
    DEFAULT_METADATA_TSV,
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    read_metadata_tsv,
    write_json,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aligned disease-90 embeddings")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_EXPORT_TSV)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_EMBED_STATS_JSON)
    args = parse_args_with_defaults(parser)

    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, checkpoint = checkpoint_embeddings_and_objects(args.checkpoint)

    missing = [node_id for node_id in objects if node_id not in metadata_map]
    if missing:
        raise ValueError(f"{len(missing)} checkpoint nodes missing from metadata, example: {missing[:5]}")
    if len(objects) != len(set(objects)):
        raise ValueError("Checkpoint object list contains duplicates")

    out_rows = []
    dim = embeddings.shape[1]
    fieldnames = ["node_id", "coding", "meaning", "depth", "top_branch_id"] + [
        f"dim_{index + 1}" for index in range(dim)
    ]
    radii = np.linalg.norm(embeddings, axis=1)
    for node_id, vector in zip(objects, embeddings):
        meta = metadata_map[node_id]
        row = {
            "node_id": node_id,
            "coding": meta["coding"],
            "meaning": meta["meaning"],
            "depth": meta["depth"],
            "top_branch_id": meta["top_branch_id"],
        }
        for index, value in enumerate(vector, start=1):
            row[f"dim_{index}"] = f"{float(value):.12g}"
        out_rows.append(row)

    write_tsv(args.out_tsv, out_rows, fieldnames)
    write_json(
        args.stats_json,
        {
            "node_count": len(objects),
            "embedding_dim": dim,
            "radius_min": float(radii.min()),
            "radius_mean": float(radii.mean()),
            "radius_max": float(radii.max()),
            "model_type": checkpoint.get("model_type"),
            "epoch": checkpoint.get("epoch"),
        },
    )
    print(f"Wrote aligned embeddings to {args.out_tsv}")
    print(f"Wrote embedding stats to {args.stats_json}")


if __name__ == "__main__":
    main()

