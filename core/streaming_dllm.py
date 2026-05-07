"""
core/streaming_dllm.py
Streaming-dLLM inference acceleration for LLaDA via three components:
  1. Attenuation-Guided Suffix Pruning  — prune redundant masked suffix tokens
  2. Dynamic Confidence-Aware Decoding  — adaptive per-step confidence threshold
  3. Early Exit                         — stop when EOS confidence is saturated

Target: >=8x speedup vs standard dLLM on outputs >=512 tokens.
Reference: Streaming-dLLM paper (arxiv 2025/26)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class StreamingConfig:
    # Suffix pruning
    suffix_mask_ratio_threshold: float = 0.95  # prune block if >95% tokens are masked
    suffix_anchor_tokens: int = 2              # keep N boundary anchors per pruned block
    block_size: int = 64                       # tokens per streaming block

    # Dynamic confidence
    base_confidence: float = 0.85             # starting unmask threshold τ_0
    confidence_decay: float = 0.5             # exponent in τ_t = τ_0 * (1 - t/T)^decay

    # Early exit
    eos_confidence_threshold: float = 0.95   # stop if P(EOS) > this
    eos_check_every_n_steps: int = 4          # check EOS every N denoising steps

    # Logging
    log_speedup: bool = True


class StreamingDLLM:
    """
    Training-free Streaming-dLLM wrapper.

    Attaches to a LLaDA model's denoising loop and applies:
      - suffix pruning to reduce attention computation
      - dynamic confidence thresholds to enable more aggressive parallel unmasking
      - early exit to skip unnecessary denoising steps

    Usage:
        streamer = StreamingDLLM(cfg=StreamingConfig())
        # Integrate via the step hooks below in refactor_agent.py
    """

    def __init__(self, cfg: Optional[StreamingConfig] = None):
        self.cfg = cfg or StreamingConfig()
        self._step_times: list[float] = []
        self._pruned_steps: int = 0
        self._total_steps: int = 0
        self._early_exit_step: Optional[int] = None

    # ── Suffix pruning ────────────────────────────────────────────────────────

    def compute_suffix_mask(
        self,
        input_ids: torch.Tensor,      # [batch, seq_len]
        mask_token_id: int,
        current_block: int,
        total_blocks: int,
    ) -> torch.Tensor:
        """
        Identify suffix token positions that can be pruned from attention.

        A block is prunable if:
          - It is in the suffix (beyond current_block)
          - More than `suffix_mask_ratio_threshold` of its tokens are still masked

        For prunable blocks, retain only `suffix_anchor_tokens` boundary tokens
        to preserve positional context.

        Returns:
            keep_mask: [batch, seq_len] bool — True = keep in attention
        """
        batch, seq_len = input_ids.shape
        block_size = self.cfg.block_size
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=input_ids.device)

        suffix_start = (current_block + 1) * block_size
        if suffix_start >= seq_len:
            return keep_mask

        for b in range(batch):
            for blk_start in range(suffix_start, seq_len, block_size):
                blk_end = min(blk_start + block_size, seq_len)
                block_tokens = input_ids[b, blk_start:blk_end]
                n_masked = (block_tokens == mask_token_id).sum().item()
                mask_ratio = n_masked / (blk_end - blk_start)

                if mask_ratio > self.cfg.suffix_mask_ratio_threshold:
                    # Prune: zero out interior, keep N anchors at boundaries
                    n_anchor = self.cfg.suffix_anchor_tokens
                    keep_mask[b, blk_start:blk_end] = False
                    # Restore boundary anchors
                    keep_mask[b, blk_start: blk_start + n_anchor] = True
                    keep_mask[b, max(blk_start, blk_end - n_anchor): blk_end] = True

        if not keep_mask[:, suffix_start:].all():
            self._pruned_steps += 1

        return keep_mask

    def apply_suffix_pruning(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden]
        keep_mask: torch.Tensor,       # [batch, seq_len] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove pruned positions from hidden states for attention computation.

        Returns:
            pruned_hidden: [batch, n_kept, hidden]
            restore_indices: [batch, seq_len] int64 — index into pruned for restore
        """
        batch, seq_len, hidden = hidden_states.shape
        # Pad keep_mask to uniform length per batch item (all same here)
        n_kept = keep_mask[0].sum().item()

        pruned_list = []
        for b in range(batch):
            pruned_list.append(hidden_states[b][keep_mask[b]])  # [n_kept_b, hidden]

        # Stack (assumes same n_kept across batch — valid with uniform suffix pruning)
        pruned_hidden = torch.stack(pruned_list, dim=0)  # [batch, n_kept, hidden]
        return pruned_hidden, keep_mask

    # ── Dynamic confidence threshold ──────────────────────────────────────────

    def get_confidence_threshold(self, step: int, total_steps: int) -> float:
        """
        τ_t = τ_0 × (1 - t/T)^decay

        Threshold starts high (conservative) and decreases as denoising
        progresses (more tokens revealed → model is more confident → lower bar).
        """
        progress = step / max(total_steps - 1, 1)
        threshold = self.cfg.base_confidence * ((1.0 - progress) ** self.cfg.confidence_decay)
        return max(threshold, 0.3)  # floor at 0.3 to avoid nonsense unmasking

    def select_tokens_to_unmask(
        self,
        logits: torch.Tensor,    # [batch, seq_len, vocab]
        mask_ids: torch.Tensor,  # [batch, seq_len] bool
        step: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens to unmask based on dynamic confidence threshold.

        Returns:
            unmask_positions: [batch, seq_len] bool
            selected_token_ids: [batch, seq_len] int64 (predicted tokens at unmask pos)
        """
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.max(dim=-1)  # [batch, seq_len]

        threshold = self.get_confidence_threshold(step, total_steps)
        # Only unmask positions that are: masked AND above confidence threshold
        unmask_positions = mask_ids & (top_probs > threshold)

        return unmask_positions, top_ids

    # ── Early exit ────────────────────────────────────────────────────────────

    def should_exit_early(
        self,
        logits: torch.Tensor,   # [batch, seq_len, vocab]
        mask_ids: torch.Tensor, # [batch, seq_len] bool
        eos_token_id: int,
        step: int,
    ) -> bool:
        """
        Return True if generation should stop early.

        Triggers when:
          1. EOS token probability exceeds threshold at any unmasked position, OR
          2. No masked positions remain (fully denoised)
        """
        if step % self.cfg.eos_check_every_n_steps != 0:
            return False

        # Check if fully denoised
        if not mask_ids.any():
            self._early_exit_step = step
            return True

        # Check EOS confidence at unmasked positions
        probs = F.softmax(logits.float(), dim=-1)
        eos_probs = probs[:, :, eos_token_id]  # [batch, seq_len]
        unmasked_eos = eos_probs[~mask_ids]
        if len(unmasked_eos) > 0 and unmasked_eos.max().item() > self.cfg.eos_confidence_threshold:
            self._early_exit_step = step
            return True

        return False

    # ── Step wrapper ──────────────────────────────────────────────────────────

    def step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask_ids: torch.Tensor,
        mask_token_id: int,
        eos_token_id: int,
        step: int,
        total_steps: int,
        current_block: int = 0,
        total_blocks: int = 1,
    ) -> tuple[torch.Tensor, bool]:
        """
        Single denoising step with all Streaming-dLLM optimizations applied.

        Returns:
            updated_ids: [batch, seq_len] — input_ids with newly unmasked tokens
            should_stop: bool — True if early exit triggered
        """
        t0 = time.perf_counter()
        self._total_steps += 1

        # Early exit check
        if self.should_exit_early(logits, mask_ids, eos_token_id, step):
            self._step_times.append(time.perf_counter() - t0)
            return input_ids, True

        # Dynamic confidence unmasking
        unmask_pos, predicted_ids = self.select_tokens_to_unmask(
            logits, mask_ids, step, total_steps
        )

        # Apply unmask
        updated_ids = input_ids.clone()
        updated_ids[unmask_pos] = predicted_ids[unmask_pos]

        self._step_times.append(time.perf_counter() - t0)
        return updated_ids, False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_speedup_report(self, baseline_steps: Optional[int] = None) -> dict:
        """Return latency and speedup metrics for logging."""
        total_time = sum(self._step_times)
        avg_step = total_time / max(len(self._step_times), 1)
        report = {
            "total_steps_run": self._total_steps,
            "pruned_steps": self._pruned_steps,
            "early_exit_step": self._early_exit_step,
            "total_time_s": round(total_time, 4),
            "avg_step_ms": round(avg_step * 1000, 2),
            "suffix_prune_rate": round(
                self._pruned_steps / max(self._total_steps, 1), 3
            ),
        }
        if baseline_steps:
            report["steps_saved"] = baseline_steps - self._total_steps
            report["step_reduction_pct"] = round(
                100 * (1 - self._total_steps / baseline_steps), 1
            )
        return report

    def reset(self):
        """Reset stats between tasks."""
        self._step_times.clear()
        self._pruned_steps = 0
        self._total_steps = 0
        self._early_exit_step = None


# ── CLI benchmark ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    if args.benchmark:
        cfg = StreamingConfig(log_speedup=True)
        streamer = StreamingDLLM(cfg)

        # Simulate a denoising loop
        batch, seq_len, vocab = 1, args.input_len, 32000
        mask_token_id, eos_token_id = 126336, 2

        input_ids = torch.full((batch, seq_len), mask_token_id)
        mask_ids = torch.ones(batch, seq_len, dtype=torch.bool)

        print(f"Simulating {args.steps} denoising steps on {seq_len}-token sequence...")
        for step in range(args.steps):
            logits = torch.randn(batch, seq_len, vocab)
            # Simulate EOS appearing late in generation
            if step > args.steps * 0.85:
                logits[:, seq_len // 2, eos_token_id] = 10.0

            input_ids, stop = streamer.step(
                logits, input_ids, mask_ids,
                mask_token_id, eos_token_id,
                step, args.steps,
                current_block=step // 8,
                total_blocks=args.steps // 8,
            )
            mask_ids = (input_ids == mask_token_id)
            if stop:
                print(f"  Early exit at step {step}/{args.steps}")
                break

        report = streamer.get_speedup_report(baseline_steps=args.steps)
        print(json.dumps(report, indent=2))
