#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from disease90_common import RESULTS_DIR, parse_args_with_defaults


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare disease-90 evaluation runs")
    parser.add_argument("eval_json", nargs="+", type=Path)
    parser.add_argument("--out-md", type=Path, default=RESULTS_DIR / "comparison_summary.md")
    args = parse_args_with_defaults(parser)

    rows = []
    for path in args.eval_json:
        metrics = load_json(path)
        name = path.stem.replace("_eval_metrics", "")
        rows.append(
            {
                "name": name,
                "recon_map": metrics["reconstruction"]["map_rank"],
                "recon_mean_rank": metrics["reconstruction"]["mean_rank"],
                "depth_spearman": metrics["depth_radius"]["spearman"],
                "parent_mean_rank": metrics["parent_ranking"]["mean_rank"],
                "ancestor_map": metrics["ancestor_ranking"]["mean_average_precision"],
                "sibling_mean": metrics["sibling_cohesion"]["siblings"]["mean"],
                "non_sibling_mean": metrics["sibling_cohesion"]["same_depth_non_siblings"]["mean"],
                "within_branch_mean": metrics["branch_separation"]["within_branch"]["mean"],
                "across_branch_mean": metrics["branch_separation"]["across_branch"]["mean"],
            }
        )

    lines = [
        "# Disease-90 run comparison",
        "",
        "| run | recon MAP | recon mean rank | depth Spearman | parent mean rank | ancestor MAP | sibling mean | same-depth non-sibling mean | within-branch mean | across-branch mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {recon_map:.4f} | {recon_mean_rank:.4f} | {depth_spearman:.4f} | {parent_mean_rank:.4f} | {ancestor_map:.4f} | {sibling_mean:.4f} | {non_sibling_mean:.4f} | {within_branch_mean:.4f} | {across_branch_mean:.4f} |".format(
                **row
            )
        )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)


if __name__ == "__main__":
    main()
