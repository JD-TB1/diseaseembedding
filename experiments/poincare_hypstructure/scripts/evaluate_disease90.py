#!/usr/bin/env python3

import argparse
from pathlib import Path

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVAL_JSON,
    DEFAULT_EVAL_MD,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    parse_args_with_defaults,
    write_json,
)
from disease90_metrics import compute_disease90_metrics, load_metadata_and_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate disease-90 Poincare+HypStructure embeddings")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--relations-csv", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_EVAL_MD)
    args = parse_args_with_defaults(parser)

    metadata_rows, relations = load_metadata_and_relations(args.metadata_tsv, args.relations_csv)
    metrics = compute_disease90_metrics(args.checkpoint, metadata_rows, relations)
    write_json(args.out_json, metrics)

    radius_structure = metrics["radius_structure"]
    lines = [
        "# Disease-90 embedding evaluation",
        "",
        f"- Reconstruction mean rank: {metrics['reconstruction']['mean_rank']:.4f}",
        f"- Reconstruction MAP: {metrics['reconstruction']['map_rank']:.4f}",
        f"- Depth/radius Spearman: {metrics['depth_radius']['spearman']:.4f}",
        f"- Depth/radius Pearson: {metrics['depth_radius']['pearson']:.4f}",
        f"- Parent mean rank: {metrics['parent_ranking']['mean_rank']:.4f}",
        f"- Ancestor MAP: {metrics['ancestor_ranking']['mean_average_precision']:.4f}",
        f"- Minimum adjacent depth gap: {radius_structure['minimum_adjacent_gap']:.6f}",
        f"- Minimum adjacent depth quantile gap: {radius_structure['minimum_adjacent_quantile_gap']:.6f}",
        f"- Positive adjacent depth quantile gaps: {radius_structure['positive_adjacent_depth_quantile_gap_count']}",
        f"- Parent-child radial violation rate: {radius_structure['parent_child_radial_violation_rate']:.4f}",
        f"- Leaf/internal radius ratio: {radius_structure['leaf_internal_radius_ratio']:.4f}",
        f"- Leaf mean radius: {radius_structure['leaf_mean_radius']:.6f}",
        f"- Mean sibling distance: {metrics['sibling_cohesion']['siblings']['mean']:.4f}",
        f"- Mean same-depth non-sibling distance: {metrics['sibling_cohesion']['same_depth_non_siblings']['mean']:.4f}",
        f"- Mean within-branch distance: {metrics['branch_separation']['within_branch']['mean']:.4f}",
        f"- Mean across-branch distance: {metrics['branch_separation']['across_branch']['mean']:.4f}",
        f"- Sibling/non-sibling distance ratio: {metrics['ratios']['sibling_to_non_sibling_mean']:.4f}",
        f"- Within/across branch ratio: {metrics['ratios']['within_branch_to_across_branch_mean']:.4f}",
        f"- Same-depth within/across branch ratio: {metrics['ratios']['same_depth_within_branch_to_across_branch_mean']:.4f}",
        f"- Angular within/across branch ratio: {metrics['ratios']['angular_within_branch_to_across_branch_mean']:.4f}",
        f"- Branch silhouette: {metrics['branch_geometry']['branch_silhouette']:.4f}",
        f"- Top-branch centroid mean distance: {metrics['branch_geometry']['top_branch_centroid_distances']['mean']:.4f}",
        f"- Gate deficit score: {metrics['gate_deficit_score']:.4f}",
        "",
        "Mean radius by depth:",
    ]
    for depth, value in radius_structure["mean_radius_by_depth"].items():
        lines.append(f"- depth {depth}: {value:.6f}")
    lines.extend(
        [
            "",
            "Adjacent depth gaps:",
        ]
    )
    for key, value in radius_structure["adjacent_depth_gaps"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(
        [
            "",
            "Adjacent depth quantile gaps:",
        ]
    )
    for key, value in radius_structure["adjacent_depth_quantile_gaps"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(
        [
            "",
            "Adjacent depth overlap rates:",
        ]
    )
    for key, value in radius_structure["depth_overlap_rates"].items():
        lines.append(f"- {key}: {value:.6f}")

    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote metrics to {args.out_json}")
    print(f"Wrote summary to {args.out_md}")


if __name__ == "__main__":
    main()
