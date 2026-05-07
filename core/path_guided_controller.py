"""
core/path_guided_controller.py
Path-Guided Unmasking Controller for LLaDA-based code refactoring.

Uses the DAWN attention graph combined with a topological sort of the
AST/import dependency graph to define a priority-ordered unmasking schedule.

Core idea: unmask LEAF nodes (imports, constants, type annotations) first,
then interior nodes (function bodies), then ROOT nodes (module-level
orchestration). This mirrors how a human expert approaches a large refactor
and forces the model to resolve dependencies before the code that uses them.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CodeNode:
    """A logical unit of code that can be independently unmasked."""
    node_id: str                    # e.g. "module_a.py::ClassFoo::method_bar"
    filename: str
    node_type: str                  # 'import', 'constant', 'class', 'function', 'module'
    start_line: int
    end_line: int
    token_span: tuple[int, int] = (-1, -1)   # (start_tok, end_tok) in prompt
    dependencies: list[str] = field(default_factory=list)   # other node_ids this depends on
    priority: int = 0               # lower = unmask earlier (computed by topo sort)
    confidence: float = 0.0         # updated each denoising step


@dataclass
class UnmaskSchedule:
    """Priority queue of CodeNodes ordered for unmasking."""
    ordered_nodes: list[CodeNode]   # topological order (index 0 = first to unmask)
    current_index: int = 0

    def next_batch(self, n: int = 4) -> list[CodeNode]:
        """Return the next N nodes to unmask."""
        batch = self.ordered_nodes[self.current_index: self.current_index + n]
        self.current_index += len(batch)
        return batch

    def remaining(self) -> int:
        return len(self.ordered_nodes) - self.current_index

    def is_complete(self) -> bool:
        return self.current_index >= len(self.ordered_nodes)


# ── AST analysis ──────────────────────────────────────────────────────────────

def extract_code_nodes(filename: str, source: str) -> list[CodeNode]:
    """
    Parse a Python source file and extract all logical CodeNodes.
    Assigns node types: import < constant < class < function < module_body
    """
    nodes: list[CodeNode] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nodes

    lines = source.splitlines()

    for node in ast.iter_child_nodes(tree):
        start = node.lineno if hasattr(node, 'lineno') else 0
        end = node.end_lineno if hasattr(node, 'end_lineno') else start

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node_type = 'import'
            name = f"import_{start}"
        elif isinstance(node, ast.Assign) and start == end:
            node_type = 'constant'
            name = f"const_{start}"
        elif isinstance(node, ast.ClassDef):
            node_type = 'class'
            name = node.name
        elif isinstance(node, ast.FunctionDef):
            node_type = 'function'
            name = node.name
        else:
            node_type = 'module_body'
            name = f"module_{start}"

        nodes.append(CodeNode(
            node_id=f"{filename}::{name}",
            filename=filename,
            node_type=node_type,
            start_line=start,
            end_line=end,
        ))

    return nodes


def build_full_dependency_graph(
    source_files: dict[str, str]
) -> tuple[list[CodeNode], dict[str, list[str]]]:
    """
    Build a full cross-file dependency graph from all source files.

    Returns:
        all_nodes: flat list of all CodeNodes across all files
        adj: {node_id: [dependent_node_ids]} — edges point FROM depender TO dependency
    """
    all_nodes: list[CodeNode] = []
    file_nodes: dict[str, list[CodeNode]] = {}
    module_map: dict[str, str] = {}   # module_name -> filename

    for fname, source in source_files.items():
        mod = os.path.splitext(os.path.basename(fname))[0]
        module_map[mod] = fname
        nodes = extract_code_nodes(fname, source)
        file_nodes[fname] = nodes
        all_nodes.extend(nodes)

    # Build node lookup
    node_by_id: dict[str, CodeNode] = {n.node_id: n for n in all_nodes}

    # Infer inter-file dependencies from import statements
    adj: dict[str, list[str]] = defaultdict(list)
    for fname, source in source_files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            dep_mod = None
            if isinstance(node, ast.Import):
                dep_mod = node.names[0].name.split('.')[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                dep_mod = node.module.split('.')[0]

            if dep_mod and dep_mod in module_map:
                dep_fname = module_map[dep_mod]
                # All nodes in fname depend on all import nodes in dep_fname
                for src_node in file_nodes.get(fname, []):
                    for dep_node in file_nodes.get(dep_fname, []):
                        if dep_node.node_type == 'import':
                            if dep_node.node_id not in adj[src_node.node_id]:
                                adj[src_node.node_id].append(dep_node.node_id)
                                src_node.dependencies.append(dep_node.node_id)

    # Intra-file: imports < constants < classes/functions < module_body
    TYPE_ORDER = {'import': 0, 'constant': 1, 'class': 2, 'function': 2, 'module_body': 3}
    for fname, nodes in file_nodes.items():
        for i, n in enumerate(nodes):
            for j, m in enumerate(nodes):
                if i != j and TYPE_ORDER[m.node_type] < TYPE_ORDER[n.node_type]:
                    if m.node_id not in adj[n.node_id]:
                        adj[n.node_id].append(m.node_id)
                        n.dependencies.append(m.node_id)

    return all_nodes, dict(adj)


# ── Topological sort ──────────────────────────────────────────────────────────

def topological_sort(
    nodes: list[CodeNode],
    adj: dict[str, list[str]],
) -> list[CodeNode]:
    """
    Kahn's algorithm topological sort on the dependency graph.
    Nodes with no dependencies come first (leaves = safe to unmask first).
    Within the same topological tier, sort by node type priority.
    """
    TYPE_PRIORITY = {'import': 0, 'constant': 1, 'class': 2, 'function': 2, 'module_body': 3}
    node_by_id = {n.node_id: n for n in nodes}

    # Compute in-degree (number of dependencies each node has)
    in_degree: dict[str, int] = {n.node_id: len(n.dependencies) for n in nodes}

    # Reverse adj: who depends on me?
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    for nid, deps in adj.items():
        for dep in deps:
            reverse_adj[dep].append(nid)

    # Start with nodes that have no dependencies
    queue = deque(
        sorted(
            [n.node_id for n in nodes if in_degree[n.node_id] == 0],
            key=lambda nid: TYPE_PRIORITY.get(node_by_id[nid].node_type, 99)
        )
    )

    sorted_nodes: list[CodeNode] = []
    priority = 0
    while queue:
        nid = queue.popleft()
        n = node_by_id[nid]
        n.priority = priority
        priority += 1
        sorted_nodes.append(n)

        # Reduce in-degree of dependents
        for dependent_id in reverse_adj[nid]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    # Append any remaining nodes (cycles — treat as lowest priority)
    visited = {n.node_id for n in sorted_nodes}
    for n in nodes:
        if n.node_id not in visited:
            n.priority = priority
            priority += 1
            sorted_nodes.append(n)

    return sorted_nodes


# ── Token span mapping ────────────────────────────────────────────────────────

def assign_token_spans(
    nodes: list[CodeNode],
    tokenizer,
    source_files: dict[str, str],
    prompt_ids: torch.Tensor,
) -> list[CodeNode]:
    """
    Map each CodeNode's line range to a token span in the prompt.
    Updates node.token_span in-place.
    """
    prompt_list = prompt_ids[0].tolist()

    for fname, source in source_files.items():
        lines = source.splitlines()
        file_nodes = [n for n in nodes if n.filename == fname]

        for node in file_nodes:
            # Extract the source lines for this node
            node_source = '\n'.join(lines[node.start_line - 1: node.end_line])
            node_ids = tokenizer(node_source, add_special_tokens=False)['input_ids']
            n = len(node_ids)

            # Search for this token subsequence in the prompt
            for i in range(len(prompt_list) - n + 1):
                if prompt_list[i: i + n] == node_ids:
                    node.token_span = (i, i + n)
                    break

    return nodes


# ── Controller ────────────────────────────────────────────────────────────────

class PathGuidedController:
    """
    Orchestrates LLaDA's denoising to follow the topological dependency order.

    Usage:
        controller = PathGuidedController(source_files, tokenizer)
        schedule = controller.build_schedule(prompt_ids)

        for step in range(num_steps):
            # Get next nodes to unmask
            batch = schedule.next_batch(n=4)
            token_spans = [n.token_span for n in batch if n.token_span[0] != -1]
            # Apply targeted unmasking at those spans only
            updated_ids = controller.apply_targeted_unmask(
                updated_ids, logits, token_spans, confidence_threshold=0.8
            )
    """

    def __init__(
        self,
        source_files: dict[str, str],
        tokenizer=None,
        nodes_per_step: int = 4,
    ):
        self.source_files = source_files
        self.tokenizer = tokenizer
        self.nodes_per_step = nodes_per_step
        self._all_nodes: list[CodeNode] = []
        self._adj: dict[str, list[str]] = {}

    def build_schedule(self, prompt_ids: torch.Tensor) -> UnmaskSchedule:
        """
        Full pipeline: parse → dependency graph → topo sort → token span mapping.
        Returns an UnmaskSchedule ready for step-by-step execution.
        """
        # 1. Extract nodes and dependency graph
        self._all_nodes, self._adj = build_full_dependency_graph(self.source_files)

        # 2. Topological sort → priority order
        sorted_nodes = topological_sort(self._all_nodes, self._adj)

        # 3. Assign token spans (if tokenizer is available)
        if self.tokenizer is not None:
            sorted_nodes = assign_token_spans(
                sorted_nodes, self.tokenizer, self.source_files, prompt_ids
            )

        return UnmaskSchedule(ordered_nodes=sorted_nodes)

    def apply_targeted_unmask(
        self,
        input_ids: torch.Tensor,         # [batch, seq_len]
        logits: torch.Tensor,            # [batch, seq_len, vocab]
        token_spans: list[tuple[int, int]],
        confidence_threshold: float = 0.8,
    ) -> torch.Tensor:
        """
        Unmask tokens only within the specified token spans,
        subject to confidence threshold. Supports batched input_ids (K_particles).
        """
        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.max(dim=-1)  # [batch, seq_len]

        updated = input_ids.clone()
        for start, end in token_spans:
            if start == -1:
                continue
            for pos in range(start, min(end, input_ids.shape[1])):
                for b in range(input_ids.shape[0]):
                    if top_probs[b, pos].item() >= confidence_threshold:
                        updated[b, pos] = top_ids[b, pos]

        return updated

    def get_schedule_summary(self, schedule: UnmaskSchedule) -> dict:
        """Return a human-readable summary of the unmasking schedule."""
        type_counts: dict[str, int] = defaultdict(int)
        for n in schedule.ordered_nodes:
            type_counts[n.node_type] += 1

        return {
            "total_nodes": len(schedule.ordered_nodes),
            "by_type": dict(type_counts),
            "first_5": [
                {"id": n.node_id, "type": n.node_type, "priority": n.priority}
                for n in schedule.ordered_nodes[:5]
            ],
            "last_5": [
                {"id": n.node_id, "type": n.node_type, "priority": n.priority}
                for n in schedule.ordered_nodes[-5:]
            ],
        }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Demo: two-file Python project with cross-file imports
    sample_files = {
        "models.py": """\
from dataclasses import dataclass

MAX_ITEMS = 100

@dataclass
class Item:
    name: str
    value: float

def create_item(name: str, value: float) -> Item:
    return Item(name=name, value=value)
""",
        "processor.py": """\
from models import Item, create_item

BATCH_SIZE = 32

class Processor:
    def __init__(self):
        self.items = []

    def add(self, name: str, value: float):
        self.items.append(create_item(name, value))

    def run(self):
        return [i.value for i in self.items]
""",
    }

    controller = PathGuidedController(source_files=sample_files)
    all_nodes, adj = build_full_dependency_graph(sample_files)
    sorted_nodes = topological_sort(all_nodes, adj)

    print("=== Path-Guided Unmasking Schedule ===")
    print(f"Total nodes: {len(sorted_nodes)}\n")
    for n in sorted_nodes:
        deps = f" (depends on: {n.dependencies})" if n.dependencies else ""
        print(f"  [{n.priority:2d}] {n.node_type:12s} | {n.node_id}{deps}")

    print("\n✅ Controller ready — topological order enforces dependency-first unmasking")
