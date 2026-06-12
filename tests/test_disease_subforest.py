from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "poincare_hypstructure" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_disease_subforest import build_prefix_subforest_metadata  # noqa: E402
from disease90_common import DATA_PATH, build_relation_rows, read_datacode_tsv  # noqa: E402


def synthetic_records() -> list[dict[str, str]]:
    return [
        {"node_id": "20", "parent_id": "0", "coding": "Chapter II", "meaning": "Chapter II Neoplasms", "selectable": "N"},
        {"node_id": "60", "parent_id": "0", "coding": "Chapter VI", "meaning": "Chapter VI Nervous", "selectable": "N"},
        {"node_id": "90", "parent_id": "0", "coding": "Chapter IX", "meaning": "Chapter IX Circulatory", "selectable": "N"},
        {"node_id": "440", "parent_id": "20", "coding": "Block C00-C14", "meaning": "C block", "selectable": "N"},
        {"node_id": "870", "parent_id": "60", "coding": "Block G00-G09", "meaning": "G block", "selectable": "N"},
        {"node_id": "1200", "parent_id": "90", "coding": "Block I10-I15", "meaning": "I block", "selectable": "N"},
        {"node_id": "590", "parent_id": "20", "coding": "Block D00-D09", "meaning": "D block", "selectable": "N"},
        {"node_id": "12070", "parent_id": "440", "coding": "C00", "meaning": "C00 target", "selectable": "Y"},
        {"node_id": "12080", "parent_id": "12070", "coding": "C000", "meaning": "C000 target", "selectable": "Y"},
        {"node_id": "31760", "parent_id": "870", "coding": "G00", "meaning": "G00 target", "selectable": "Y"},
        {"node_id": "43400", "parent_id": "1200", "coding": "I10", "meaning": "I10 target", "selectable": "Y"},
        {"node_id": "17660", "parent_id": "590", "coding": "D00", "meaning": "D00 excluded", "selectable": "Y"},
    ]


def test_individual_prefix_uses_natural_chapter_root_and_block_branches() -> None:
    metadata_rows, ancestors, summary = build_prefix_subforest_metadata(
        synthetic_records(),
        ["C"],
        root_mode="individual",
    )
    by_id = {row["node_id"]: row for row in metadata_rows}
    relations = build_relation_rows(metadata_rows, ancestors, mode="direct")

    assert metadata_rows[0]["node_id"] == "20"
    assert by_id["20"]["parent_id"] == "0"
    assert by_id["440"]["top_branch_id"] == "440"
    assert by_id["12080"]["top_branch_id"] == "440"
    assert by_id["12070"]["is_target_code"] == "Y"
    assert by_id["440"]["is_target_code"] == "N"
    assert "17660" not in by_id
    assert summary["root_mode"] == "individual"
    assert summary["synthetic_node_count"] == 0
    assert summary["target_code_count"] == 2
    assert summary["C_count"] == 2
    assert summary["training_node_count"] == 4
    assert summary["direct_edge_count"] == 3
    assert summary["top_branch_count"] == 1
    assert len(relations) == 3


def test_combined_prefixes_add_synthetic_root_and_chapter_branches() -> None:
    metadata_rows, ancestors, summary = build_prefix_subforest_metadata(
        synthetic_records(),
        ["C", "G", "I"],
        root_mode="combined",
        synthetic_root_id="CGI_ROOT",
        synthetic_root_code="CGI_ROOT",
    )
    by_id = {row["node_id"]: row for row in metadata_rows}
    relations = build_relation_rows(metadata_rows, ancestors, mode="direct")

    assert metadata_rows[0]["node_id"] == "CGI_ROOT"
    assert by_id["20"]["parent_id"] == "CGI_ROOT"
    assert by_id["60"]["parent_id"] == "CGI_ROOT"
    assert by_id["90"]["parent_id"] == "CGI_ROOT"
    assert by_id["12080"]["top_branch_id"] == "20"
    assert by_id["31760"]["top_branch_id"] == "60"
    assert by_id["43400"]["top_branch_id"] == "90"
    assert "17660" not in by_id
    assert summary["root_mode"] == "combined"
    assert summary["target_code_count"] == 4
    assert summary["C_count"] == 2
    assert summary["G_count"] == 1
    assert summary["I_count"] == 1
    assert summary["training_node_count"] == 11
    assert summary["direct_edge_count"] == 10
    assert summary["top_branch_count"] == 3
    assert len(relations) == 10


def test_real_prefix_subforest_counts_match_expected_contract() -> None:
    records = read_datacode_tsv(DATA_PATH)

    cases = [
        (["C"], "individual", 559, 575, 574, 15),
        (["G"], "individual", 396, 408, 407, 11),
        (["I"], "individual", 475, 486, 485, 10),
        (["C", "G", "I"], "combined", 1430, 1470, 1469, 3),
    ]
    for prefixes, root_mode, target_count, node_count, edge_count, top_branch_count in cases:
        metadata_rows, ancestors, summary = build_prefix_subforest_metadata(
            records,
            prefixes,
            root_mode=root_mode,
            synthetic_root_id="CGI_ROOT" if root_mode == "combined" else None,
        )
        relations = build_relation_rows(metadata_rows, ancestors, mode="direct")
        target_rows = [row for row in metadata_rows if row["is_target_code"] == "Y"]

        assert summary["target_code_count"] == target_count
        assert summary["training_node_count"] == node_count
        assert summary["direct_edge_count"] == edge_count
        assert summary["top_branch_count"] == top_branch_count
        assert len(relations) == edge_count
        assert len(target_rows) == target_count
        assert not any(row["coding"].startswith("D") for row in target_rows)
        for row in target_rows:
            assert row["coding"].startswith(tuple(prefixes))
