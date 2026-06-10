from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "poincare_hypstructure" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_cg_subforest import CG_ROOT_ID, build_cg_subforest_metadata  # noqa: E402
from disease90_common import DATA_PATH, build_relation_rows, read_datacode_tsv  # noqa: E402


def test_cg_builder_filters_codes_and_adds_synthetic_root() -> None:
    records = [
        {"node_id": "20", "parent_id": "0", "coding": "Chapter II", "meaning": "Chapter II Neoplasms", "selectable": "N"},
        {"node_id": "60", "parent_id": "0", "coding": "Chapter VI", "meaning": "Chapter VI Nervous", "selectable": "N"},
        {"node_id": "440", "parent_id": "20", "coding": "Block C00-C14", "meaning": "C block", "selectable": "N"},
        {"node_id": "870", "parent_id": "60", "coding": "Block G00-G09", "meaning": "G block", "selectable": "N"},
        {"node_id": "590", "parent_id": "20", "coding": "Block D00-D09", "meaning": "D block", "selectable": "N"},
        {"node_id": "12070", "parent_id": "440", "coding": "C00", "meaning": "C00 target", "selectable": "Y"},
        {"node_id": "12080", "parent_id": "12070", "coding": "C000", "meaning": "C000 target", "selectable": "Y"},
        {"node_id": "17660", "parent_id": "590", "coding": "D00", "meaning": "D00 excluded", "selectable": "Y"},
        {"node_id": "31760", "parent_id": "870", "coding": "G00", "meaning": "G00 target", "selectable": "Y"},
    ]

    metadata_rows, ancestors, summary = build_cg_subforest_metadata(records)
    by_id = {row["node_id"]: row for row in metadata_rows}

    assert metadata_rows[0]["node_id"] == CG_ROOT_ID
    assert summary["target_code_count"] == 3
    assert summary["C_count"] == 2
    assert summary["G_count"] == 1
    assert summary["training_node_count"] == 8
    assert by_id["20"]["parent_id"] == CG_ROOT_ID
    assert by_id["60"]["parent_id"] == CG_ROOT_ID
    assert by_id["12080"]["top_branch_id"] == "20"
    assert by_id["31760"]["top_branch_id"] == "60"
    assert by_id["20"]["is_target_code"] == "N"
    assert "17660" not in by_id

    relations = build_relation_rows(metadata_rows, ancestors, mode="direct")

    assert len(relations) == len(metadata_rows) - 1


def test_real_cg_builder_matches_expected_contract() -> None:
    metadata_rows, ancestors, summary = build_cg_subforest_metadata(read_datacode_tsv(DATA_PATH))
    relations = build_relation_rows(metadata_rows, ancestors, mode="direct")
    target_rows = [row for row in metadata_rows if row["is_target_code"] == "Y"]

    assert summary["target_code_count"] == 955
    assert summary["C_count"] == 559
    assert summary["G_count"] == 396
    assert summary["real_node_count"] == 983
    assert summary["training_node_count"] == 984
    assert summary["direct_edge_count"] == 983
    assert len(relations) == 983
    assert not any(row["coding"].startswith("D") for row in target_rows)
    assert {row["top_branch_code"] for row in metadata_rows if row["depth"] == "1"} == {"Chapter II", "Chapter VI"}
