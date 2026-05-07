"""
core/dawn_decoder.py
Plug-and-play DAWN (Dependency-Aware fast iNference) wrapper for LLaDA.

Three components (paper: lizhuo-luo/DAWN on GitHub):
  1. Dependency graph: attention maps -> sparse directed graph
  2. Anchor-guided decoding: high-confidence positions unlock dependents
  3. Conflict-based scheduling: greedy independent set -> parallel unmask

Code-specific extension: inject AST-derived import edges as hard priors.
"""
from __future__ import annotations
import ast, os
from typing import Optional
import numpy as np
import torch


def build_ast_import_graph(source_files: dict[str, str]) -> dict[str, list[str]]:
    """Extract inter-module import dependencies from Python source files.
    Returns {filename: [dependency_filenames]}."""
    module_map = {os.path.splitext(os.path.basename(f))[0]: f for f in source_files}
    graph: dict[str, list[str]] = {f: [] for f in source_files}

    for fname, source in source_files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            dep_mod = None
            if isinstance(node, ast.Import):
                dep_mod = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                dep_mod = node.module.split(".")[0]
            if dep_mod and dep_mod in module_map:
                dep_file = module_map[dep_mod]
                if dep_file != fname and dep_file not in graph[fname]:
                    graph[fname].append(dep_file)
    return graph


class DAWNDecoder:
    """
    DAWN decoder with AST-prior injection for code refactoring.

    Usage:
        decoder = DAWNDecoder(source_files={"a.py": src_a, "b.py": src_b},
                              tokenizer=tokenizer)
        # Pass to LLaDA generate() via dawn_decoder= kwarg
    """

    def __init__(
        self,
        source_files: Optional[dict[str, str]] = None,
        tokenizer=None,
        top_k_attention: int = 32,
        anchor_confidence: float = 0.85,
        conflict_threshold: float = 0.6,
        inject_ast_priors: bool = True,
    ):
        self.source_files = source_files or {}
        self.tokenizer = tokenizer
        self.top_k_attention = top_k_attention
        self.anchor_confidence = anchor_confidence
        self.conflict_threshold = conflict_threshold
        self._dep_matrix: Optional[np.ndarray] = None
        self._token_spans: dict[str, tuple[int, int]] = {}
        self._ast_graph: dict[str, list[str]] = {}

        if source_files and inject_ast_priors:
            self._ast_graph = build_ast_import_graph(source_files)

    # ── Dependency matrix ─────────────────────────────────────────────────────

    def build_dependency_matrix(self, prompt_ids: torch.Tensor) -> np.ndarray:
        """Build [seq_len x seq_len] hard-prior matrix from AST import edges."""
        seq_len = prompt_ids.shape[1]
        dep_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

        if not self.source_files or not self.tokenizer:
            return dep_matrix

        # Map each file's tokens to a span in the prompt
        cursor, prompt_list = 0, prompt_ids[0].tolist()
        for fname, source in self.source_files.items():
            file_ids = self.tokenizer(source, add_special_tokens=False)["input_ids"]
            n = len(file_ids)
            for i in range(cursor, len(prompt_list) - n + 1):
                if prompt_list[i: i + n] == file_ids:
                    self._token_spans[fname] = (i, i + n)
                    cursor = i + n
                    break
            else:
                self._token_spans[fname] = (-1, -1)

        # Inject import edges as hard priors
        for fname, deps in self._ast_graph.items():
            fs, fe = self._token_spans.get(fname, (-1, -1))
            if fs == -1:
                continue
            for dep in deps:
                ds, de = self._token_spans.get(dep, (-1, -1))
                if ds != -1:
                    dep_matrix[fs:fe, ds:de] = 1.0

        self._dep_matrix = dep_matrix
        return dep_matrix

    # ── Attention graph ───────────────────────────────────────────────────────

    def extract_attention_graph(
        self, attention_maps: list[torch.Tensor], seq_len: int
    ) -> np.ndarray:
        """Average multi-layer attention -> top-k sparse graph, merged with AST priors."""
        stacked = torch.stack([a[0].mean(0) for a in attention_maps])
        avg = stacked.mean(0).float().cpu().numpy()

        k = min(self.top_k_attention, seq_len)
        sparse = np.zeros_like(avg)
        for i in range(seq_len):
            idx = np.argpartition(avg[i], -k)[-k:]
            sparse[i, idx] = avg[i, idx]

        row_sums = sparse.sum(1, keepdims=True).clip(min=1e-8)
        sparse /= row_sums

        if self._dep_matrix is not None:
            sparse = np.maximum(sparse, self._dep_matrix)
        return sparse

    # ── Scheduling ────────────────────────────────────────────────────────────

    def select_unmask_positions(
        self,
        logits: torch.Tensor,
        mask_ids: torch.Tensor,
        attention_maps: list[torch.Tensor],
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Return boolean tensor of positions to unmask this diffusion step."""
        batch_size, seq_len, _ = logits.shape
        confidence = torch.softmax(logits.float(), -1).max(-1).values

        progress = step / max(total_steps - 1, 1)
        threshold = self.anchor_confidence * (1.0 - 0.3 * progress)

        unmask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)

        for b in range(batch_size):
            conf = confidence[b].cpu().numpy()
            masked = mask_ids[b].cpu().numpy()
            is_anchor = (~masked) | (conf > threshold)
            anchors = np.where(is_anchor)[0]

            if len(anchors) == 0:
                # Fallback: unmask single highest-confidence masked position
                m_idx = np.where(masked)[0]
                if len(m_idx):
                    unmask[b, m_idx[conf[m_idx].argmax()]] = True
                continue

            if attention_maps is not None:
                attn = self.extract_attention_graph([a[b:b+1] for a in attention_maps], seq_len)
            else:
                # If the model does not support attention extraction (like LLaDA),
                # fallback purely to our AST dependency matrix
                attn = self._dep_matrix if self._dep_matrix is not None else np.zeros((seq_len, seq_len))
            
            m_idx = np.where(masked)[0]
            scores = attn[np.ix_(anchors, m_idx)].max(0) if len(anchors) > 0 and attn.shape == (seq_len, seq_len) else np.zeros(len(m_idx))
            induced = m_idx[scores > self.conflict_threshold]

            for idx in self._greedy_independent_set(induced, attn):
                unmask[b, idx] = True

        return unmask

    @staticmethod
    def _greedy_independent_set(candidates: np.ndarray, adj: np.ndarray,
                                 threshold: float = 0.6) -> list[int]:
        selected, excluded = [], set()
        for idx in candidates:
            if idx in excluded:
                continue
            selected.append(int(idx))
            for other in candidates:
                if other != idx and other not in excluded:
                    if adj[idx, other] > threshold and adj[other, idx] > threshold:
                        excluded.add(other)
        return selected

    def get_stats(self) -> dict:
        return {
            "ast_edges": sum(len(v) for v in self._ast_graph.values()),
            "token_spans_mapped": sum(1 for v in self._token_spans.values() if v[0] != -1),
            "top_k": self.top_k_attention,
        }
