"""
core/hybrid_orchestrator.py
Manages the Multi-Model Hybrid Orchestration pipeline for Phase 4.
Lead Architect (LLaDA) -> Blueprint Extraction -> Staff Engineer (DeepSeek AR)
"""

import time
import re
import torch
import gc
from typing import Optional
from dataclasses import dataclass

from core.model_loader import load_model, ModelConfig

@dataclass
class HybridResult:
    final_output: str
    blueprint_text: str
    llada_latency: float
    deepseek_latency: float
    total_latency: float


class HybridOrchestrator:
    """
    Solves the Speed vs Quality trade-off.
    1. Loads LLaDA (DLLM) to generate a high-level structural blueprint using topological unmasking.
    2. Halts early and converts unmasked regions into <FIM_HOLE> markers.
    3. Unloads LLaDA and loads DeepSeek-Coder-6.7B (AR).
    4. DeepSeek fills the holes to provide functional low-level logic.
    """
    
    def __init__(self, llada_cfg: ModelConfig, deepseek_cfg: ModelConfig):
        self.llada_cfg = llada_cfg
        self.deepseek_cfg = deepseek_cfg

    def _free_memory(self):
        """Force cleanup to prevent 24GB L4 GPU OOM when swapping models."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_pipeline(
        self,
        prompt: str,
        source_files: dict[str, str],
        k_particles: int = 1,
    ) -> HybridResult:
        
        total_t0 = time.perf_counter()
        
        # ─── PHASE 1: LEAD ARCHITECT (LLaDA 1.5) ──────────────────────────────
        print("\n[Hybrid] Loading LLaDA 1.5 (Lead Architect)...")
        llada_loader = load_model(self.llada_cfg)
        
        from core.path_guided_controller import PathGuidedController
        path_controller = PathGuidedController(
            source_files=source_files,
            tokenizer=llada_loader.tokenizer
        )
        
        # We halt at 40% of the total diffusion steps. 
        # By this time, Kahn's Topological Sort ensures imports, classes, and signatures are unmasked.
        total_steps = self.llada_cfg.num_diffusion_steps
        halt_step = int(0.4 * total_steps)
        print(f"[Hybrid] Running DLLM topological unmasking (halting at step {halt_step}/{total_steps})...")
        
        raw_blueprint, llada_latency = llada_loader.generate(
            prompt=prompt,
            path_controller=path_controller,
            k_particles=k_particles,
            halt_at_step=halt_step,
        )
        
        # Unload DLLM to free VRAM for DeepSeek
        print("[Hybrid] DLLM Blueprint complete. Freeing VRAM...")
        del llada_loader.model
        del llada_loader
        self._free_memory()
        
        # ─── PHASE 2: BLUEPRINT EXTRACTION ────────────────────────────────────
        # LLaDA leaves unresolved tokens as `<|mask|>`. 
        # We collapse contiguous masks into a single instruction hole.
        print("[Hybrid] Extracting Structural Blueprint...")
        
        # Regex collapse multiple masks into a single FILL_ME block
        collapsed_blueprint = re.sub(
            r'(<\|mdm_mask\|>)+', 
            '\n        # <FILL_ME>\n        pass\n', 
            raw_blueprint
        )
        
        # Fallback if no masks were left (unlikely at 40% steps, but possible on tiny files)
        if '<FILL_ME>' not in collapsed_blueprint:
            print("[Hybrid] Warning: No <FILL_ME> holes generated. Blueprint is fully unmasked.")
            return HybridResult(
                final_output=collapsed_blueprint,
                blueprint_text=collapsed_blueprint,
                llada_latency=llada_latency,
                deepseek_latency=0.0,
                total_latency=time.perf_counter() - total_t0
            )

        # ─── PHASE 3: STAFF ENGINEER (DeepSeek AR) ────────────────────────────
        print("\n[Hybrid] Loading DeepSeek-Coder-6.7B (Staff Engineer)...")
        deepseek_loader = load_model(self.deepseek_cfg)
        
        # We prompt DeepSeek to perform Infilling on the blueprint
        fim_prompt = (
            "You are an expert Staff Engineer.\n"
            "The following is a high-level structural blueprint for a refactored codebase. "
            "Your task is to replace all instances of `# <FILL_ME>` and `pass` with the correct, "
            "low-level implementation logic. Do not change the function signatures or imports.\n\n"
            "```python\n"
            f"{collapsed_blueprint}\n"
            "```\n\n"
            "Provide the completed code below:\n"
            "```python\n"
        )
        
        print("[Hybrid] Running AR Functional Infilling...")
        final_output, deepseek_latency = deepseek_loader.generate(
            prompt=fim_prompt,
            max_new_tokens=1500,  # Allow enough tokens for all holes
        )
        
        # Cleanup
        del deepseek_loader.model
        del deepseek_loader
        self._free_memory()
        
        total_latency = time.perf_counter() - total_t0
        print(f"[Hybrid] Pipeline complete. Total Latency: {total_latency:.2f}s")
        
        return HybridResult(
            final_output=final_output,
            blueprint_text=collapsed_blueprint,
            llada_latency=llada_latency,
            deepseek_latency=deepseek_latency,
            total_latency=total_latency
        )
