"""
core/model_loader.py
Unified loader for LLaDA 1.5 (dLLM) and DeepSeek-Coder-6.7B (AR baseline).
Exposes a common generate() interface for the eval harness.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_type: Literal["llada", "deepseek"]
    quantize: Literal["none", "int8", "int4"] = "none"
    device: str = "cuda"
    max_new_tokens: int = 512
    # LLaDA-specific
    num_diffusion_steps: int = 64
    # DeepSeek-specific
    temperature: float = 0.2
    do_sample: bool = True


# HuggingFace model IDs
_MODEL_IDS = {
    "llada":    "GSAI-ML/LLaDA-8B-Instruct",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


# ── Quantization helper ───────────────────────────────────────────────────────

def _bnb_config(quantize: str) -> Optional[BitsAndBytesConfig]:
    if quantize == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ── Model wrappers ────────────────────────────────────────────────────────────

class BaseRefactorModel:
    """Common interface for all model backends."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def generate(self, prompt: str, **kwargs) -> tuple[str, float]:
        """Returns (generated_text, latency_seconds)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.cfg.model_type


class LLaDaModel(BaseRefactorModel):
    """
    LLaDA 1.5 discrete diffusion model wrapper.
    Uses the official generate() function from the LLaDA repo, which
    implements masked diffusion denoising.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        model_id = _MODEL_IDS["llada"]

        print(f"[LLaDA] Loading {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        bnb = _bnb_config(cfg.quantize)
        load_kwargs = dict(
            trust_remote_code=True,
        )
        if bnb:
            load_kwargs["quantization_config"] = bnb
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        # LLaDA uses a custom model class — loaded via trust_remote_code
        from transformers import AutoModel, PreTrainedModel
        
        # STRONG MONKEY-PATCH: The custom LLaDA code explicitly sets its tied weights to None, 
        # which breaks `transformers>=4.45`. We will completely disable the failing method 
        # in the base class before loading.
        PreTrainedModel.mark_tied_weights_as_initialized = lambda self, *args, **kwargs: None

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        
        if not bnb:
            self.model = self.model.to(cfg.device)
            
        self.model.eval()
        device_info = getattr(self.model, 'hf_device_map', self.model.device)
        print(f"[LLaDA] Loaded. Device: {device_info}")

    def generate(self, prompt: str, **kwargs) -> tuple[str, float]:
        """
        Run masked diffusion generation.
        Imports LLaDA's generate() from the cloned repo at ~/repos/LLaDA.
        """
        import sys
        llada_path = os.path.expanduser("~/repos/LLaDA")
        if llada_path not in sys.path:
            sys.path.insert(0, llada_path)
        from generate import generate as llada_generate  # LLaDA's own generate fn

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        gen_length = kwargs.get("max_new_tokens", self.cfg.max_new_tokens)
        steps = kwargs.get("num_diffusion_steps", self.cfg.num_diffusion_steps)
        K_particles = kwargs.get("k_particles", 1)  # Phase 4: POKE-SMC Test-Time Scaling
        halt_at_step = kwargs.get("halt_at_step", -1)
        
        dawn = kwargs.get("dawn_decoder")
        path_ctrl = kwargs.get("path_controller")
        
        t0 = time.perf_counter()
        with torch.inference_mode():
            if dawn or path_ctrl:
                # Custom DAWN / Path-Guided Diffusion Loop
                mask_id = 126336 # LLaDA mask token ID
                device = inputs.input_ids.device
                
                gen_ids = torch.full((1, gen_length), mask_id, dtype=torch.long, device=device)
                current_ids = torch.cat([inputs.input_ids, gen_ids], dim=1)
                
                # Expand to multiple particles for Sequential Monte Carlo
                if K_particles > 1:
                    current_ids = current_ids.expand(K_particles, -1).clone()
                path_ll = torch.zeros(K_particles, device=device)
                
                if dawn:
                    dawn.build_dependency_matrix(inputs.input_ids)
                    
                schedule = None
                if path_ctrl:
                    schedule = path_ctrl.build_schedule(inputs.input_ids)
                
                for step in range(steps):
                    # LLaDA doesn't support output_attentions yet, so we just use the AST priors
                    outputs = self.model(current_ids)
                    logits = outputs.logits[:, -gen_length:]
                    attentions = None
                    
                    mask_bool = (current_ids[:, -gen_length:] == mask_id)
                    if not mask_bool.any():
                        break
                        
                    if halt_at_step != -1 and step >= halt_at_step:
                        break
                        
                    # Phase 4: POKE-SMC Resampling (Kinetic Ensembles)
                    if K_particles > 1 and step > 0 and step % 10 == 0:
                        weights = torch.softmax(path_ll, dim=0)
                        indices = torch.multinomial(weights, K_particles, replacement=True)
                        current_ids = current_ids[indices].clone()
                        path_ll = path_ll[indices].clone()
                        mask_bool = mask_bool[indices]
                        logits = logits[indices]
                        
                    # Phase 3: Path-Guided Topological Unmasking
                    # During first 20% of steps, bias unmasking toward leaf AST nodes.
                    # Fall through to DAWN/naive for remaining masked tokens.
                    if path_ctrl and step < int(0.2 * steps) and not schedule.is_complete():
                        batch = schedule.next_batch(n=path_ctrl.nodes_per_step)
                        # Only apply targeted unmask to nodes that have valid token spans
                        valid_spans = [
                            n.token_span for n in batch
                            if n.token_span[0] != -1 and n.token_span[1] <= gen_length
                        ]
                        if valid_spans:
                            current_ids[:, -gen_length:] = path_ctrl.apply_targeted_unmask(
                                current_ids[:, -gen_length:], logits, valid_spans,
                                confidence_threshold=0.75
                            )
                        # Fall through to DAWN/naive for remaining masked tokens

                    if dawn:
                        unmask_bool = dawn.select_unmask_positions(
                            logits, mask_bool, attentions, step, steps
                        )
                    else:
                        # Fallback to naive confidence unmasking if DAWN is missing
                        confidence = torch.softmax(logits.float(), -1).max(-1).values
                        threshold = 0.85 * (1.0 - 0.3 * (step / steps))
                        unmask_bool = mask_bool & (confidence > threshold)
                        for b in range(K_particles):
                            if not unmask_bool[b].any() and mask_bool[b].any():
                                m_idx = torch.where(mask_bool[b])[0]
                                best_idx = m_idx[confidence[b, m_idx].argmax()]
                                unmask_bool[b, best_idx] = True
                            
                    preds = logits.argmax(-1)
                    
                    # Accumulate Path Log-Likelihood for unmasked tokens
                    if K_particles > 1:
                        log_probs = torch.log_softmax(logits.float(), dim=-1)
                        chosen_log_probs = log_probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
                        path_ll += (chosen_log_probs * unmask_bool).sum(dim=1)

                    current_ids[:, -gen_length:][unmask_bool] = preds[unmask_bool]
                    
                # Select the globally most coherent particle
                best_idx = path_ll.argmax().item() if K_particles > 1 else 0
                output_ids = current_ids[best_idx:best_idx+1, -gen_length:]
            else:
                # Naive Loop
                output_ids = llada_generate(
                    self.model,
                    inputs.input_ids,
                    steps=steps,
                    gen_length=gen_length,
                )
        latency = time.perf_counter() - t0
        
        # Decode the output
        if isinstance(output_ids, torch.Tensor):
            keep_specials = kwargs.get("halt_at_step", -1) != -1
            out = self.tokenizer.decode(output_ids[0], skip_special_tokens=not keep_specials)
            if keep_specials:
                # Remove bos/eos but leave <|mask|> for the orchestrator
                out = out.replace('<s>', '').replace('</s>', '').replace('<|endoftext|>', '')
        else:
            out = str(output_ids)
            
        return out, latency


class DeepSeekCoderModel(BaseRefactorModel):
    """
    DeepSeek-Coder-6.7B-Instruct autoregressive baseline.
    Standard HuggingFace causal LM generation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        model_id = _MODEL_IDS["deepseek"]

        print(f"[DeepSeek] Loading {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        bnb = _bnb_config(cfg.quantize)
        load_kwargs = dict(device_map="auto")
        if bnb:
            load_kwargs["quantization_config"] = bnb
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        device_info = getattr(self.model, 'hf_device_map', self.model.device)
        print(f"[DeepSeek] Loaded. Device: {device_info}")

    def generate(self, prompt: str, **kwargs) -> tuple[str, float]:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.cfg.device)

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", self.cfg.max_new_tokens),
            temperature=self.cfg.temperature,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, latency


# ── Factory ───────────────────────────────────────────────────────────────────

def load_model(cfg: ModelConfig) -> BaseRefactorModel:
    """
    Factory function. Usage:
        cfg = ModelConfig(model_type="llada", quantize="int4")
        model = load_model(cfg)
        output, latency = model.generate("Refactor this class to functional style:\n...")
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Run this on the GCP VM with an L4 or A100."
        )

    if cfg.model_type == "llada":
        return LLaDaModel(cfg)
    elif cfg.model_type == "deepseek":
        return DeepSeekCoderModel(cfg)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Use 'llada' or 'deepseek'.")


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model loader smoke test")
    parser.add_argument("--model", choices=["llada", "deepseek"], default="deepseek")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="int4")
    args = parser.parse_args()

    cfg = ModelConfig(model_type=args.model, quantize=args.quantize, max_new_tokens=128)
    model = load_model(cfg)

    prompt = (
        "Refactor the following Python class to use pure functions and dataclasses:\n\n"
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "    def increment(self):\n"
        "        self.count += 1\n"
        "    def get(self):\n"
        "        return self.count\n"
    )

    print(f"\n{'='*60}")
    print(f"Model: {args.model} | Quantize: {args.quantize}")
    print(f"Prompt:\n{prompt}")
    print("="*60)

    output, latency = model.generate(prompt)
    print(f"\nOutput:\n{output}")
    print(f"\nLatency: {latency:.2f}s")
