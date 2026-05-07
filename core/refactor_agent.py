"""
core/refactor_agent.py
End-to-end orchestration pipeline for the DLLM refactoring agent.

Pipeline:
  Input:  (source files dict, refactoring instruction string)
  Step 1: Parse AST → build dependency graph
  Step 2: Build DAWN decoder with AST priors
  Step 3: Build PathGuidedController → topological schedule
  Step 4: Construct masked prompt
  Step 5: Run inference (LLaDA or DeepSeek) with DAWN + Streaming-dLLM
  Step 6: Parse output → per-file diffs
  Step 7: Validate (run tests + AST structural distance)
  Output: RefactorResult (refactored files, metrics, latency)
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from core.model_loader import BaseRefactorModel, ModelConfig, load_model
from core.dawn_decoder import DAWNDecoder
from core.streaming_dllm import StreamingDLLM, StreamingConfig
from core.path_guided_controller import PathGuidedController


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RefactorResult:
    task_id: str
    model_name: str
    decoder: str                          # 'none' | 'dawn' | 'dawn+streaming'
    original_files: dict[str, str]
    refactored_files: dict[str, str]
    instruction: str

    # Metrics
    pass_at_1: Optional[bool] = None
    ast_structural_distance: Optional[float] = None
    cross_file_coherence: Optional[float] = None
    latency_seconds: float = 0.0
    speedup_vs_baseline: Optional[float] = None

    # Diagnostics
    dawn_stats: dict = field(default_factory=dict)
    streaming_stats: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model_name,
            "decoder": self.decoder,
            "pass_at_1": self.pass_at_1,
            "ast_structural_distance": self.ast_structural_distance,
            "cross_file_coherence": self.cross_file_coherence,
            "latency_seconds": round(self.latency_seconds, 3),
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "dawn_stats": self.dawn_stats,
            "streaming_stats": self.streaming_stats,
            "error": self.error,
        }


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python refactoring agent. You will be given one or more Python \
source files and a refactoring instruction. Output ONLY the refactored files, \
each preceded by a line: ### FILE: <filename>
Do not include any explanation. Preserve all existing behavior.\
"""

def build_refactor_prompt(
    source_files: dict[str, str],
    instruction: str,
) -> str:
    """Construct the prompt fed to the model."""
    parts = [SYSTEM_PROMPT, f"\n## Refactoring Instruction\n{instruction}\n\n## Source Files\n"]
    for fname, source in source_files.items():
        parts.append(f"### FILE: {fname}\n```python\n{source}\n```\n")
    parts.append("## Refactored Output\n")
    return "\n".join(parts)


# ── Output parsing ────────────────────────────────────────────────────────────

def parse_refactored_output(
    raw_output: str,
    original_filenames: list[str],
) -> dict[str, str]:
    """
    Parse model output into a {filename: source_code} dict.
    Expects sections delimited by: ### FILE: <filename>
    Falls back to returning the full output under the first filename.
    """
    result: dict[str, str] = {}
    current_file = None
    current_lines: list[str] = []

    # Clean up byte-level BPE artifacts often leaked by LlamaTokenizerFast
    raw_output = raw_output.replace("Ġ", " ").replace("Ċ", "\n")

    for line in raw_output.splitlines():
        if line.startswith("### FILE:"):
            if current_file:
                src = "\n".join(current_lines).strip()
                # Strip markdown code fences if present
                src = src.replace("```python", "").replace("```", "").strip()
                result[current_file] = src
            current_file = line.replace("### FILE:", "").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_file:
        src = "\n".join(current_lines).strip()
        src = src.replace("```python", "").replace("```", "").strip()
        result[current_file] = src

    # Fallback: no delimiters found
    if not result and original_filenames:
        src = raw_output.strip()
        # If the model just output a big markdown block, extract it
        import re
        matches = re.findall(r'```python\n(.*?)\n```', src, re.DOTALL)
        if matches:
            src = "\n".join(matches)
        else:
            # Try generic code block
            matches = re.findall(r'```\n(.*?)\n```', src, re.DOTALL)
            if matches:
                src = "\n".join(matches)
        
        src = src.replace("```python", "").replace("```", "").strip()
        result[original_filenames[0]] = src

    return result


# ── Evaluation helpers ────────────────────────────────────────────────────────

def compute_ast_structural_distance(
    original: str, refactored: str
) -> Optional[float]:
    """
    Compute normalized tree-edit distance between original and refactored ASTs.
    Uses the ZSS (Zhang-Shasha) algorithm via the `zss` package.
    Returns value in [0, 1]: 0 = identical structure, 1 = maximally different.
    """
    try:
        import zss

        def ast_to_zss(node):
            label = type(node).__name__
            children = [ast_to_zss(c) for c in ast.iter_child_nodes(node)]
            return zss.Node(label, children)

        orig_tree = ast_to_zss(ast.parse(original))
        refact_tree = ast_to_zss(ast.parse(refactored))
        dist = zss.simple_distance(orig_tree, refact_tree)

        # Normalize by sum of node counts
        orig_size = sum(1 for _ in ast.walk(ast.parse(original)))
        refact_size = sum(1 for _ in ast.walk(ast.parse(refactored)))
        norm = dist / max(orig_size + refact_size, 1)
        return round(min(norm, 1.0), 4)
    except Exception:
        return None


def compute_cross_file_coherence(
    original_files: dict[str, str],
    refactored_files: dict[str, str]
) -> float:
    """
    Measure what fraction of inter-file import references are still valid
    after refactoring.
    
    Uses Python's native AST module to calculate:
    Coherence Score = 1 - (StaleReferences / TotalReferences).
    """
    import ast
    
    def get_exported_symbols(source: str) -> set[str]:
        if not source: return set()
        try:
            tree = ast.parse(source)
            return {
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            }
        except SyntaxError:
            return set()

    def get_function_calls(source: str) -> set[str]:
        if not source: return set()
        try:
            tree = ast.parse(source)
            calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
            return calls
        except SyntaxError:
            return []

    try:
        # Step 1: Registry of all deleted or renamed exported symbols
        deleted_symbols = set()
        for fname, orig_src in original_files.items():
            orig_sym = get_exported_symbols(orig_src)
            refact_src = refactored_files.get(fname, "")
            refact_sym = get_exported_symbols(refact_src)
            deleted_symbols.update(orig_sym - refact_sym)

        # Step 2: Query for stale references across all refactored files
        stale_references = 0
        total_references = 0

        for fname, refact_src in refactored_files.items():
            calls = get_function_calls(refact_src)
            total_references += len(calls)
            for call in calls:
                if call in deleted_symbols:
                    stale_references += 1

        if total_references == 0:
            return 1.0

        score = 1.0 - (stale_references / total_references)
        return round(score, 4)

    except Exception as e:
        print(f"[WARN] AST coherence failed: {e}")
        return 0.0


def run_tests(
    refactored_files: dict[str, str],
    test_dir: Optional[str] = None,
) -> bool:
    """
    Write refactored files to a temp directory, run pytest, return pass/fail.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write refactored source files
        for fname, source in refactored_files.items():
            fpath = os.path.join(tmpdir, os.path.basename(fname))
            with open(fpath, 'w') as f:
                f.write(source)

        # Copy test files if provided
        if test_dir and os.path.isdir(test_dir):
            for tf in Path(test_dir).glob("test_*.py"):
                dest = os.path.join(tmpdir, tf.name)
                with open(tf) as src, open(dest, 'w') as dst:
                    dst.write(src.read())

        result = subprocess.run(
            ["python", "-m", "pytest", tmpdir, "-q", "--tb=no", "--no-header"],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0


# ── Main agent ────────────────────────────────────────────────────────────────

class RefactorAgent:
    """
    Orchestrates the full refactoring pipeline.

    Modes:
      - model_type='deepseek', decoder='none'  → AR baseline
      - model_type='llada',    decoder='none'  → LLaDA naive
      - model_type='llada',    decoder='dawn'  → LLaDA + DAWN
      - model_type='llada',    decoder='dawn+streaming' → full stack
      - decoder='hybrid'                       → LLaDA -> FIM Blueprint -> DeepSeek
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        decoder: str = 'none',          # 'none' | 'dawn' | 'dawn+streaming' | 'hybrid'
        results_dir: str = 'results',
    ):
        self.model_cfg = model_cfg
        self.decoder_mode = decoder
        self.results_dir = results_dir
        self.model: Optional[BaseRefactorModel] = None
        os.makedirs(results_dir, exist_ok=True)

    def _lazy_load_model(self):
        if self.model is None:
            self.model = load_model(self.model_cfg)

    def refactor(
        self,
        task_id: str,
        source_files: dict[str, str],
        instruction: str,
        test_dir: Optional[str] = None,
        baseline_latency: Optional[float] = None,
        k_particles: int = 1,
    ) -> RefactorResult:
        """Run the full refactoring pipeline for one task."""
        if self.decoder_mode != 'hybrid':
            self._lazy_load_model()


        result = RefactorResult(
            task_id=task_id,
            model_name=self.model_cfg.model_type,
            decoder=self.decoder_mode,
            original_files=source_files,
            refactored_files={},
            instruction=instruction,
        )

        try:
            # ── Step 1: Build prompt ───────────────────────────────────────
            prompt = build_refactor_prompt(source_files, instruction)

            if self.decoder_mode == 'hybrid':
                from core.hybrid_orchestrator import HybridOrchestrator
                llada_cfg = ModelConfig(model_type='llada', quantize=self.model_cfg.quantize, max_new_tokens=self.model_cfg.max_new_tokens)
                deepseek_cfg = ModelConfig(model_type='deepseek', quantize=self.model_cfg.quantize, max_new_tokens=1500)
                
                orchestrator = HybridOrchestrator(llada_cfg, deepseek_cfg)
                hybrid_result = orchestrator.run_pipeline(
                    prompt=prompt,
                    source_files=source_files,
                    k_particles=k_particles,
                )
                
                raw_output = hybrid_result.final_output
                latency = hybrid_result.total_latency
                result.latency_seconds = latency
            else:
                # ── Step 2: Set up decoders ────────────────────────────────────
                gen_kwargs: dict = {}
    
                if 'dawn' in self.decoder_mode and self.model_cfg.model_type == 'llada':
                    dawn = DAWNDecoder(
                        source_files=source_files,
                        tokenizer=self.model.tokenizer,
                        inject_ast_priors=True,
                    )
                    gen_kwargs['dawn_decoder'] = dawn
                    result.dawn_stats = dawn.get_stats()
    
                if 'streaming' in self.decoder_mode and self.model_cfg.model_type == 'llada':
                    streamer = StreamingDLLM(StreamingConfig())
                    gen_kwargs['streaming'] = streamer
    
                if self.model_cfg.model_type == 'llada' and 'path' in self.decoder_mode:
                    controller = PathGuidedController(
                        source_files=source_files,
                        tokenizer=self.model.tokenizer,
                    )
                    gen_kwargs['path_controller'] = controller
    
                # ── Step 3: Generate ───────────────────────────────────────────
                raw_output, latency = self.model.generate(
                    prompt,
                    max_new_tokens=self.model_cfg.max_new_tokens,
                    k_particles=k_particles,
                    **gen_kwargs,
                )
                result.latency_seconds = latency
    
                if 'streaming' in self.decoder_mode:
                    result.streaming_stats = streamer.get_speedup_report(
                        baseline_steps=self.model_cfg.num_diffusion_steps
                    )
    
                if baseline_latency:
                    result.speedup_vs_baseline = round(baseline_latency / latency, 2)

            # ── Step 4: Parse output ───────────────────────────────────────
            result.refactored_files = parse_refactored_output(
                raw_output, list(source_files.keys())
            )

            # ── Step 5: Evaluate ───────────────────────────────────────────
            # AST structural distance (per file, averaged)
            distances = []
            for fname in source_files:
                if fname in result.refactored_files:
                    d = compute_ast_structural_distance(
                        source_files[fname], result.refactored_files[fname]
                    )
                    if d is not None:
                        distances.append(d)
            result.ast_structural_distance = (
                round(sum(distances) / len(distances), 4) if distances else None
            )

            # Cross-file coherence
            result.cross_file_coherence = compute_cross_file_coherence(
                source_files,
                result.refactored_files
            )

            # Pass@1 (run tests if test_dir provided)
            if test_dir:
                result.pass_at_1 = run_tests(result.refactored_files, test_dir)

        except Exception as e:
            import traceback
            result.error = traceback.format_exc()

        # ── Step 6: Persist result ─────────────────────────────────────────
        self._save_result(result)
        return result

    def _save_result(self, result: RefactorResult):
        run_dir = os.path.join(self.results_dir, "runs", result.model_name + "_" + result.decoder)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f"{result.task_id}.json")
        with open(out_path, 'w') as f:
            json.dump(result.to_json(), f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single refactoring task")
    parser.add_argument("--model", choices=["llada", "deepseek"], default="deepseek")
    parser.add_argument("--decoder", choices=["none", "dawn", "dawn+streaming", "path+dawn", "hybrid"],
                        default="none")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--task-id", default="smoke_test")
    parser.add_argument("--k-particles", type=int, default=1)
    args = parser.parse_args()

    cfg = ModelConfig(
        model_type=args.model,
        quantize=args.quantize,
        max_new_tokens=512,
    )
    agent = RefactorAgent(model_cfg=cfg, decoder=args.decoder)

    # Smoke test: OOP → functional
    source_files = {
        "counter.py": """\
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def decrement(self):
        self.count -= 1

    def get(self):
        return self.count

    def reset(self):
        self.count = 0
"""
    }
    instruction = (
        "Refactor this class to use pure functions and a dataclass. "
        "Replace all methods with standalone functions that take and return state."
    )

    print(f"Running: {args.model} + decoder={args.decoder}")
    result = agent.refactor(
        task_id=args.task_id,
        source_files=source_files,
        instruction=instruction,
        k_particles=args.k_particles,
    )

    print(f"\nLatency      : {result.latency_seconds:.2f}s")
    print(f"AST Distance : {result.ast_structural_distance}")
    print(f"Coherence    : {result.cross_file_coherence}")
    print(f"Error        : {result.error}")
    print(f"\nRefactored output:\n{list(result.refactored_files.values())[0] if result.refactored_files else 'N/A'}")
