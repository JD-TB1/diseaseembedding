#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import colorsys
import html
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_TSV = ROOT / "data" / "datacode-19.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "visualizations"

REQUIRED_COLUMNS = {"coding", "meaning", "node_id", "parent_id", "selectable"}
ROUTE_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
    "#2f6b99",
    "#d37222",
    "#b63f40",
    "#5f9894",
    "#47813f",
    "#c9ab39",
    "#8e5d8b",
    "#d97f8b",
    "#7c5d4d",
    "#8f8681",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the disease forest or a selected subtree from datacode-19.tsv."
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        default=DEFAULT_INPUT_TSV,
        help="Path to the input TSV file.",
    )
    parser.add_argument(
        "--root-id",
        default="0",
        help="Root node id to visualize. Use 0 for the synthetic forest root.",
    )
    parser.add_argument(
        "--mode",
        choices=("html", "graphviz"),
        default="html",
        help="Output mode.",
    )
    parser.add_argument(
        "--layout",
        choices=("collapsible", "radial", "topdown", "leftright"),
        default="collapsible",
        help="Layout strategy. Use collapsible for HTML and radial/topdown/leftright for Graphviz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path. Use .html for HTML or .dot/.svg/.png/.pdf for Graphviz.",
    )
    parser.add_argument(
        "--title",
        default="Disease tree",
        help="Title shown in the output.",
    )
    parser.add_argument(
        "--initial-open-depth",
        type=int,
        default=1,
        help="For HTML output, open nodes through this depth.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional depth cap relative to the selected root.",
    )
    parser.add_argument(
        "--max-meaning-chars",
        type=int,
        default=72,
        help="Maximum meaning length shown inside node labels.",
    )
    parser.add_argument(
        "--graphviz-node-limit",
        type=int,
        default=2000,
        help="Guardrail for static Graphviz output.",
    )
    parser.add_argument(
        "--color-by",
        choices=("depth", "route"),
        default="depth",
        help="Color nodes by depth shell or by route under the selected root.",
    )
    parser.add_argument(
        "--focus-node-id",
        action="append",
        default=[],
        help="Keep the root-to-node path for this node id. Repeat the flag to keep several paths.",
    )
    parser.add_argument(
        "--auto-focus-by-route",
        type=int,
        default=0,
        help="Automatically keep up to N representative deepest leaf paths, one per first-level route under the selected root.",
    )
    parser.add_argument(
        "--collapse-omitted",
        action="store_true",
        help="For Graphviz output, collapse omitted sibling subtrees into dashed ellipsis nodes.",
    )
    return parser.parse_args()


def read_records(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input TSV: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV is missing required columns: {sorted(missing)}")

        records: dict[str, dict[str, str]] = {}
        for row in reader:
            node_id = (row["node_id"] or "").strip()
            if not node_id:
                continue
            records[node_id] = {
                "coding": (row["coding"] or "").strip(),
                "meaning": (row["meaning"] or "").strip(),
                "node_id": node_id,
                "parent_id": (row["parent_id"] or "").strip(),
                "selectable": (row["selectable"] or "").strip(),
            }
    return records


def build_children(records: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, row in records.items():
        children[row["parent_id"]].append(node_id)

    def sort_key(node_id: str) -> tuple[str, str, str]:
        row = records[node_id]
        return (row["coding"], row["meaning"], row["node_id"])

    for node_ids in children.values():
        node_ids.sort(key=sort_key)
    return children


def ensure_root(
    root_id: str,
    records: dict[str, dict[str, str]],
    children: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    if root_id in records:
        return records
    if root_id in children:
        records = dict(records)
        records[root_id] = {
            "coding": "ROOT" if root_id == "0" else root_id,
            "meaning": "Synthetic root" if root_id == "0" else "Synthetic root",
            "node_id": root_id,
            "parent_id": "",
            "selectable": "N",
        }
        return records
    raise ValueError(f"Root id {root_id!r} is neither a real node nor a parent in the TSV.")


def collect_subtree(
    root_id: str,
    children: dict[str, list[str]],
    max_depth: int | None,
) -> tuple[list[str], dict[str, int]]:
    ordered: list[str] = []
    depth_map: dict[str, int] = {}
    stack: list[tuple[str, int]] = [(root_id, 0)]
    seen: set[str] = set()
    while stack:
        node_id, depth = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
        depth_map[node_id] = depth
        if max_depth is not None and depth >= max_depth:
            continue
        for child_id in reversed(children.get(node_id, [])):
            stack.append((child_id, depth + 1))
    return ordered, depth_map


def compute_leaf_count(node_id: str, children: dict[str, list[str]], allowed: set[str]) -> int:
    child_ids = [child_id for child_id in children.get(node_id, []) if child_id in allowed]
    if not child_ids:
        return 1
    return sum(compute_leaf_count(child_id, children, allowed) for child_id in child_ids)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


def depth_color(depth: int, max_depth: int) -> str:
    scale = 0 if max_depth <= 0 else depth / max_depth
    hue = 215 - int(135 * scale)
    saturation = 65
    lightness = 95 - int(32 * scale)
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)
    return "#{:02x}{:02x}{:02x}".format(int(red * 255), int(green * 255), int(blue * 255))


def node_summary(
    row: dict[str, str],
    depth: int,
    child_count: int,
    leaf_count: int,
    max_meaning_chars: int,
) -> str:
    coding = row["coding"] or row["node_id"]
    meaning = truncate(row["meaning"], max_meaning_chars)
    selectable = row["selectable"] or "?"
    parts = [
        f"<span class=\"node-code\">{html.escape(coding)}</span>",
        f"<span class=\"node-meaning\">{html.escape(meaning)}</span>",
        f"<span class=\"node-meta\">id={html.escape(row['node_id'])} | depth={depth} | selectable={html.escape(selectable)} | children={child_count} | leaves={leaf_count}</span>",
    ]
    return " ".join(parts)


def render_html(
    records: dict[str, dict[str, str]],
    children: dict[str, list[str]],
    ordered_nodes: list[str],
    depth_map: dict[str, int],
    root_id: str,
    title: str,
    initial_open_depth: int,
    max_meaning_chars: int,
    output_path: Path,
) -> None:
    allowed = set(ordered_nodes)
    max_depth = max(depth_map.values()) if depth_map else 0
    selectable_count = sum(1 for node_id in ordered_nodes if records[node_id]["selectable"] == "Y")
    leaf_count = sum(1 for node_id in ordered_nodes if not [child for child in children.get(node_id, []) if child in allowed])
    edge_count = sum(1 for node_id in ordered_nodes if node_id != root_id)

    def render_node(node_id: str) -> str:
        row = records[node_id]
        depth = depth_map[node_id]
        child_ids = [child_id for child_id in children.get(node_id, []) if child_id in allowed]
        leaf_subtree_count = compute_leaf_count(node_id, children, allowed)
        content = node_summary(row, depth, len(child_ids), leaf_subtree_count, max_meaning_chars)
        search_blob = " ".join(
            part.lower()
            for part in (
                row["coding"],
                row["meaning"],
                row["node_id"],
                row["parent_id"],
            )
            if part
        )
        color = depth_color(depth, max_depth)
        if child_ids:
            open_attr = " open" if depth <= initial_open_depth else ""
            child_html = "".join(render_node(child_id) for child_id in child_ids)
            return (
                f"<li class=\"tree-node depth-{depth}\" data-search=\"{html.escape(search_blob)}\">"
                f"<details{open_attr}>"
                f"<summary style=\"background:{color}\">{content}</summary>"
                f"<ul>{child_html}</ul>"
                f"</details>"
                f"</li>"
            )
        return (
            f"<li class=\"tree-node tree-leaf depth-{depth}\" data-search=\"{html.escape(search_blob)}\">"
            f"<div class=\"leaf-row\" style=\"background:{color}\">{content}</div>"
            f"</li>"
        )

    tree_html = render_node(root_id)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #f6f8fb;
      color: #18212d;
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 32px;
      line-height: 1.15;
    }}
    .lede {{
      margin: 0 0 20px;
      color: #536173;
      font-size: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 0 0 24px;
    }}
    .stat-card {{
      background: white;
      border: 1px solid #dce4ef;
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 4px 14px rgba(24, 33, 45, 0.05);
    }}
    .stat-label {{
      display: block;
      color: #5a6676;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .tree-shell {{
      background: white;
      border: 1px solid #dce4ef;
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 6px 20px rgba(24, 33, 45, 0.06);
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 16px;
      align-items: center;
    }}
    .controls input {{
      min-width: 280px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #cdd8e6;
      font-size: 14px;
    }}
    .controls button {{
      padding: 10px 12px;
      border: 1px solid #cdd8e6;
      border-radius: 10px;
      background: #f8fbff;
      cursor: pointer;
    }}
    ul {{
      list-style: none;
      margin: 0;
      padding-left: 22px;
      position: relative;
    }}
    ul::before {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 8px;
      width: 1px;
      background: #dbe3ef;
    }}
    li {{
      margin: 6px 0;
      position: relative;
    }}
    li::before {{
      content: "";
      position: absolute;
      top: 18px;
      left: -14px;
      width: 14px;
      height: 1px;
      background: #dbe3ef;
    }}
    details > summary {{
      list-style: none;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(100, 120, 140, 0.12);
      cursor: pointer;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: baseline;
    }}
    details > summary::-webkit-details-marker {{
      display: none;
    }}
    .leaf-row {{
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(100, 120, 140, 0.12);
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: baseline;
    }}
    .node-code {{
      font-weight: 700;
      font-size: 15px;
    }}
    .node-meaning {{
      font-size: 14px;
    }}
    .node-meta {{
      color: #5b6674;
      font-size: 12px;
    }}
    .hidden {{
      display: none !important;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p class="lede">Interactive collapsible view of the selected disease tree. This layout works better than a flat same-level node-link plot when the tree is very wide.</p>
    <div class="stats">
      <div class="stat-card"><span class="stat-label">Root</span><span class="stat-value">{html.escape(root_id)}</span></div>
      <div class="stat-card"><span class="stat-label">Nodes</span><span class="stat-value">{len(ordered_nodes)}</span></div>
      <div class="stat-card"><span class="stat-label">Edges</span><span class="stat-value">{edge_count}</span></div>
      <div class="stat-card"><span class="stat-label">Leaves</span><span class="stat-value">{leaf_count}</span></div>
      <div class="stat-card"><span class="stat-label">Selectable</span><span class="stat-value">{selectable_count}</span></div>
      <div class="stat-card"><span class="stat-label">Max depth</span><span class="stat-value">{max_depth}</span></div>
    </div>
    <div class="tree-shell">
      <div class="controls">
        <input id="tree-filter" type="search" placeholder="Filter by code, meaning, node id, or parent id">
        <button type="button" onclick="setOpenState(true)">Expand all</button>
        <button type="button" onclick="setOpenState(false)">Collapse all</button>
      </div>
      <ul id="tree-root">{tree_html}</ul>
    </div>
  </div>
  <script>
    function setOpenState(isOpen) {{
      document.querySelectorAll("details").forEach((element) => {{
        element.open = isOpen;
      }});
    }}

    function applyFilter(query) {{
      const normalized = query.trim().toLowerCase();
      const nodes = Array.from(document.querySelectorAll("li.tree-node"));
      if (!normalized) {{
        nodes.forEach((node) => node.classList.remove("hidden"));
        return;
      }}

      function visit(node) {{
        const ownMatch = (node.dataset.search || "").includes(normalized);
        const childNodes = Array.from(node.querySelectorAll(":scope > details > ul > li.tree-node"));
        let descendantMatch = false;
        childNodes.forEach((childNode) => {{
          if (visit(childNode)) {{
            descendantMatch = true;
          }}
        }});
        const visible = ownMatch || descendantMatch;
        node.classList.toggle("hidden", !visible);
        const detail = node.querySelector(":scope > details");
        if (detail && descendantMatch) {{
          detail.open = true;
        }}
        return visible;
      }}

      const rootNode = document.querySelector("#tree-root > li.tree-node");
      if (rootNode) {{
        visit(rootNode);
      }}
    }}

    document.getElementById("tree-filter").addEventListener("input", (event) => {{
      applyFilter(event.target.value);
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\"", "\\\"")


def build_parent_map(records: dict[str, dict[str, str]]) -> dict[str, str]:
    return {node_id: row["parent_id"] for node_id, row in records.items()}


def route_anchor_for_node(
    node_id: str,
    root_id: str,
    parent_map: dict[str, str],
) -> str:
    if node_id == root_id:
        return root_id
    current = node_id
    while True:
        parent_id = parent_map.get(current, "")
        if parent_id == root_id:
            return current
        if not parent_id or parent_id == current:
            return current
        current = parent_id


def assign_route_colors(
    records: dict[str, dict[str, str]],
    ordered_nodes: list[str],
    root_id: str,
    parent_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    route_anchors = sorted(
        {
            route_anchor_for_node(node_id, root_id, parent_map)
            for node_id in ordered_nodes
            if node_id != root_id
        },
        key=lambda node_id: (
            records[node_id]["coding"],
            records[node_id]["meaning"],
            records[node_id]["node_id"],
        ),
    )
    route_color_map: dict[str, str] = {root_id: "#eef2f7"}
    for index, anchor_id in enumerate(route_anchors):
        route_color_map[anchor_id] = ROUTE_PALETTE[index % len(ROUTE_PALETTE)]

    node_color_map: dict[str, str] = {}
    for node_id in ordered_nodes:
        anchor_id = route_anchor_for_node(node_id, root_id, parent_map)
        node_color_map[node_id] = route_color_map.get(anchor_id, "#d7dee8")
    return node_color_map, route_color_map


def edge_color_for_child(
    child_id: str,
    root_id: str,
    route_color_map: dict[str, str],
    parent_map: dict[str, str],
) -> str:
    anchor_id = route_anchor_for_node(child_id, root_id, parent_map)
    return route_color_map.get(anchor_id, "#b8c4d6")


def subtree_size_map(
    root_id: str,
    children: dict[str, list[str]],
    allowed: set[str],
) -> dict[str, int]:
    sizes: dict[str, int] = {}

    def visit(node_id: str) -> int:
        total = 1
        for child_id in children.get(node_id, []):
            if child_id not in allowed:
                continue
            total += visit(child_id)
        sizes[node_id] = total
        return total

    visit(root_id)
    return sizes


def path_to_root(node_id: str, root_id: str, parent_map: dict[str, str]) -> list[str]:
    path: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        path.append(current)
        if current == root_id:
            return list(reversed(path))
        current = parent_map.get(current, "")
    raise ValueError(f"Node {node_id!r} is not reachable from root {root_id!r}.")


def choose_auto_focus_nodes(
    records: dict[str, dict[str, str]],
    ordered_nodes: list[str],
    root_id: str,
    parent_map: dict[str, str],
    children: dict[str, list[str]],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    allowed = set(ordered_nodes)
    route_to_leaf: dict[str, str] = {}
    for node_id in ordered_nodes:
        if node_id == root_id:
            continue
        child_ids = [child_id for child_id in children.get(node_id, []) if child_id in allowed]
        if child_ids:
            continue
        anchor_id = route_anchor_for_node(node_id, root_id, parent_map)
        current_best = route_to_leaf.get(anchor_id)
        if current_best is None:
            route_to_leaf[anchor_id] = node_id
            continue
        current_best_depth = len(path_to_root(current_best, root_id, parent_map))
        candidate_depth = len(path_to_root(node_id, root_id, parent_map))
        if candidate_depth > current_best_depth:
            route_to_leaf[anchor_id] = node_id
            continue
        if candidate_depth == current_best_depth:
            candidate_key = (
                records[node_id]["coding"],
                records[node_id]["meaning"],
                records[node_id]["node_id"],
            )
            best_key = (
                records[current_best]["coding"],
                records[current_best]["meaning"],
                records[current_best]["node_id"],
            )
            if candidate_key < best_key:
                route_to_leaf[anchor_id] = node_id

    selected = sorted(
        route_to_leaf.items(),
        key=lambda item: (
            records[item[0]]["coding"],
            records[item[0]]["meaning"],
            records[item[0]]["node_id"],
        ),
    )
    return [leaf_id for _, leaf_id in selected[:limit]]


def build_focus_keep_set(
    root_id: str,
    focus_node_ids: list[str],
    parent_map: dict[str, str],
) -> set[str]:
    keep: set[str] = {root_id}
    for focus_node_id in focus_node_ids:
        keep.update(path_to_root(focus_node_id, root_id, parent_map))
    return keep


def render_graphviz(
    records: dict[str, dict[str, str]],
    children: dict[str, list[str]],
    ordered_nodes: list[str],
    depth_map: dict[str, int],
    root_id: str,
    title: str,
    max_meaning_chars: int,
    layout: str,
    color_by: str,
    focus_node_ids: list[str],
    collapse_omitted: bool,
    output_path: Path,
    graphviz_node_limit: int,
) -> None:
    if len(ordered_nodes) > graphviz_node_limit:
        raise ValueError(
            f"Static Graphviz rendering is capped at {graphviz_node_limit} nodes; "
            f"selected tree has {len(ordered_nodes)} nodes. Use HTML for the full forest or raise --graphviz-node-limit."
        )

    max_depth = max(depth_map.values()) if depth_map else 0
    parent_map = build_parent_map(records)
    node_route_colors, route_color_map = assign_route_colors(records, ordered_nodes, root_id, parent_map)
    allowed = set(ordered_nodes)
    focus_keep = build_focus_keep_set(root_id, focus_node_ids, parent_map) if focus_node_ids else allowed
    subtree_sizes = subtree_size_map(root_id, children, allowed)
    graph_lines = [
        "digraph disease_tree {",
        "  graph [overlap=false, splines=true, pad=0.4, nodesep=0.35, ranksep=0.55, bgcolor=\"white\", labelloc=\"t\"];",
        f"  label=\"{dot_escape(title)}\";",
        "  node [shape=box, style=\"rounded,filled\", color=\"#4f5d75\", penwidth=0.8, fontname=\"Helvetica\", fontsize=10, margin=\"0.08,0.05\"];",
        "  edge [color=\"#b8c4d6\", penwidth=0.7, arrowsize=0.5];",
    ]
    if layout == "leftright":
        graph_lines.append("  rankdir=LR;")
    elif layout == "topdown":
        graph_lines.append("  rankdir=TB;")

    display_nodes = ordered_nodes if not collapse_omitted or not focus_node_ids else [node_id for node_id in ordered_nodes if node_id in focus_keep]

    for node_id in display_nodes:
        row = records[node_id]
        depth = depth_map[node_id]
        meaning = truncate(row["meaning"], max_meaning_chars)
        label_lines = [row["coding"] or row["node_id"]]
        if meaning:
            label_lines.append(meaning)
        label_lines.append(f"id={row['node_id']} | d={depth}")
        fill = depth_color(depth, max_depth) if color_by == "depth" else node_route_colors[node_id]
        graph_lines.append(
            f"  \"{dot_escape(node_id)}\" [label=\"{dot_escape(chr(10).join(label_lines))}\", fillcolor=\"{fill}\"];"
        )

    for parent_id in display_nodes:
        child_ids = [child_id for child_id in children.get(parent_id, []) if child_id in allowed]
        kept_child_ids = child_ids if not collapse_omitted or not focus_node_ids else [child_id for child_id in child_ids if child_id in focus_keep]
        omitted_child_ids = [] if not collapse_omitted or not focus_node_ids else [child_id for child_id in child_ids if child_id not in focus_keep]

        for child_id in kept_child_ids:
            if child_id not in display_nodes:
                continue
            if color_by == "route":
                edge_color = edge_color_for_child(child_id, root_id, route_color_map, parent_map)
                graph_lines.append(
                    f"  \"{dot_escape(parent_id)}\" -> \"{dot_escape(child_id)}\" [color=\"{edge_color}\", penwidth=1.0];"
                )
            else:
                graph_lines.append(f"  \"{dot_escape(parent_id)}\" -> \"{dot_escape(child_id)}\";")

        if omitted_child_ids:
            hidden_branch_count = len(omitted_child_ids)
            hidden_node_count = sum(subtree_sizes[child_id] for child_id in omitted_child_ids)
            sample_labels = ", ".join((records[child_id]["coding"] or child_id) for child_id in omitted_child_ids[:3])
            if len(omitted_child_ids) > 3:
                sample_labels += ", …"
            ellipsis_id = f"ellipsis__{parent_id}"
            ellipsis_label = f"...\\n{hidden_branch_count} hidden branches\\n{hidden_node_count} hidden nodes"
            if sample_labels:
                ellipsis_label += f"\\n{sample_labels}"
            graph_lines.append(
                f"  \"{dot_escape(ellipsis_id)}\" [shape=ellipse, style=\"dashed,filled\", fillcolor=\"#f1f3f6\", "
                f"color=\"#9aa4b2\", label=\"{dot_escape(ellipsis_label)}\", fontsize=9];"
            )
            graph_lines.append(
                f"  \"{dot_escape(parent_id)}\" -> \"{dot_escape(ellipsis_id)}\" [style=dashed, color=\"#9aa4b2\"];"
            )

    if root_id == "0":
        graph_lines.append("  {rank=source; \"0\";}")

    if color_by == "route":
        route_items = [anchor_id for anchor_id in route_color_map if anchor_id != root_id]
        if len(route_items) <= 16:
            graph_lines.append("  subgraph cluster_legend {")
            graph_lines.append("    label=\"Route legend\";")
            graph_lines.append("    color=\"#d7dee8\";")
            graph_lines.append("    style=\"rounded\";")
            graph_lines.append("    fontsize=11;")
            graph_lines.append("    fontname=\"Helvetica\";")
            for index, anchor_id in enumerate(route_items):
                route_row = records[anchor_id]
                legend_label = route_row["coding"] or route_row["node_id"]
                if route_row["meaning"]:
                    legend_label = f"{legend_label}: {truncate(route_row['meaning'], 34)}"
                graph_lines.append(
                    f"    legend_{index} [shape=box, style=\"rounded,filled\", fillcolor=\"{route_color_map[anchor_id]}\", "
                    f"label=\"{dot_escape(legend_label)}\", fontsize=10];"
                )
            graph_lines.append("  }")

    graph_lines.append("}")
    dot_text = "\n".join(graph_lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".dot":
        output_path.write_text(dot_text, encoding="utf-8")
        return

    output_format = output_path.suffix.lower().lstrip(".")
    if output_format not in {"svg", "png", "pdf"}:
        raise ValueError(f"Unsupported Graphviz output suffix: {output_path.suffix}")

    engine = "twopi" if layout == "radial" else "dot"
    subprocess.run(
        [engine, f"-T{output_format}", "-o", str(output_path)],
        input=dot_text,
        text=True,
        check=True,
    )


def main() -> int:
    args = parse_args()
    if args.mode == "html" and args.layout != "collapsible":
        raise ValueError("HTML mode currently supports only --layout collapsible")
    if args.mode == "graphviz" and args.layout == "collapsible":
        raise ValueError("Graphviz mode requires --layout radial, topdown, or leftright")

    records = read_records(args.input_tsv)
    children = build_children(records)
    records = ensure_root(args.root_id, records, children)
    if args.root_id not in children and args.root_id not in records:
        raise ValueError(f"Root id {args.root_id!r} not found in TSV")

    ordered_nodes, depth_map = collect_subtree(args.root_id, children, args.max_depth)
    if not ordered_nodes:
        raise ValueError(f"No nodes found under root {args.root_id!r}")

    parent_map = build_parent_map(records)
    auto_focus_nodes = choose_auto_focus_nodes(
        records=records,
        ordered_nodes=ordered_nodes,
        root_id=args.root_id,
        parent_map=parent_map,
        children=children,
        limit=args.auto_focus_by_route,
    )
    focus_node_ids = list(dict.fromkeys(args.focus_node_id + auto_focus_nodes))
    if focus_node_ids:
        allowed = set(ordered_nodes)
        missing_focus = [node_id for node_id in focus_node_ids if node_id not in allowed]
        if missing_focus:
            raise ValueError(
                f"Focus nodes are not inside the selected tree rooted at {args.root_id}: {missing_focus[:5]}"
            )

    if args.mode == "html":
        render_html(
            records=records,
            children=children,
            ordered_nodes=ordered_nodes,
            depth_map=depth_map,
            root_id=args.root_id,
            title=args.title,
            initial_open_depth=args.initial_open_depth,
            max_meaning_chars=args.max_meaning_chars,
            output_path=args.output,
        )
    else:
        render_graphviz(
            records=records,
            children=children,
            ordered_nodes=ordered_nodes,
            depth_map=depth_map,
            root_id=args.root_id,
            title=args.title,
            max_meaning_chars=args.max_meaning_chars,
            layout=args.layout,
            color_by=args.color_by,
            focus_node_ids=focus_node_ids,
            collapse_omitted=args.collapse_omitted,
            output_path=args.output,
            graphviz_node_limit=args.graphviz_node_limit,
        )

    print(f"Wrote {args.mode} visualization to {args.output}")
    print(f"Root id: {args.root_id}")
    print(f"Nodes in selection: {len(ordered_nodes)}")
    print(f"Max depth in selection: {max(depth_map.values()) if depth_map else 0}")
    if focus_node_ids:
        print(f"Focus nodes: {', '.join(focus_node_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
