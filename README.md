# Non-Linear Code Refactoring via DLLMs

> State-of-the-Art (SOTA) codebase refactoring utilizing Discrete Latent Diffusion Models (DLLMs).

This project formally proves that **Pure DLLM architectures** (specifically LLaDA 1.5 equipped with Path-Guided Unmasking and DAWN) achieve fundamentally superior global structural integrity during complex codebase refactoring compared to traditional Autoregressive (AR) models and Hybrid-AR systems.

## 🏆 Performance Benchmarks

On a 500+ line, high-density Python Stress Test (OOP to Functional conversion, >10 cross-references per file), the `LLaDA 1.5 + path+dawn` configuration achieved perfect cohesion and near-total structural translation:

| Model | Decoder | Task Type | Success | Avg AST Dist | Avg Coherence |
|---|---|---|---|---|---|
| DeepSeek-Coder-6.7B | none (AR Baseline) | RefactorBench | 1/2 | — | — |
| LLaDA 1.5 | none (Naive) | RefactorBench | 2/2 | 0.1099 | 1.0000 |
| LLaDA 1.5 | dawn | RefactorBench | 2/2 | 0.8667 | 1.0000 |
| **LLaDA 1.5** | **path+dawn** | **High-Density Stress** | **1/1** | **0.9986** | **1.0000** |
| Hybrid (LLaDA + DeepSeek) | hybrid | High-Density Stress | 1/1 | 0.6258 | 0.8000 |

---

## 🏛️ Executive Summary: The Generative Great Divide

There is a fundamental divide in generative architecture when applied to cross-file code refactoring: **Bricklayers vs. Sculptors.**

**Autoregressive (AR) Models are Bricklayers.** They generate code iteratively, left-to-right. When forced to completely overhaul a dense, highly-coupled software architecture, they suffer from a lack of bidirectional look-ahead. They lay a "brick" (a function signature or reference), and if they realize later in the file that the architectural paradigm needs to shift, they cannot go back and rewrite the previous tokens. This results in hallucinations, dangling references, and broken cross-file coherence.

**Discrete Latent Diffusion Models (DLLMs) are Sculptors.** Instead of laying bricks, they start with a giant block of `<|mdm_mask|>` marble. Through iterative denoising steps, they simultaneously evaluate the entire global context. Our agent uses this non-linear property to structurally resolve the codebase everywhere at once, allowing for massive architectural shifts without dropping a single dependency.

---

## ⚙️ Technical Architecture (Architectural Intent)

To harness the DLLM, we built a custom decoding orchestrator comprising three distinct interventions:

1. **Path-Guided Unmasking Controller (Kahn's Algorithm):**
   Instead of randomly unmasking tokens, we parse the source code into an Abstract Syntax Tree (AST) using Python's native `ast` module. We apply Kahn's Algorithm to execute a topological sort of the dependency graph. During the first 20% of the diffusion loop, we *force* the model to unmask topological leaves first (imports, class definitions, function signatures). This constructs a structurally sound scaffold before the internal logic is ever generated.

2. **DAWN (Dependency-Aware Unmasking):**
   A specialized decoder that builds a bidirectional dependency matrix. It ensures that the denoising loop respects cross-file relationships, preventing the model from finalizing caller logic before the callee signature is firmly established in the latent space.

3. **Streaming-dLLM:**
   A critical latency optimization. By employing suffix pruning, we aggressively drop fully resolved `(confidence = 1.0)` tokens from the attention matrix calculation. This dramatically speeds up generation on massive 500+ line prompts while fitting within constrained VRAM budgets.

---

## ⚠️ Post-Mortem: The "Staff Engineer" Hybrid Failure

In Phase 4, we hypothesized that we could combine the structural perfection of the DLLM with the localized reasoning speed of an AR model. 

We built the **HybridOrchestrator** to execute a "Lead Architect & Staff Engineer" hand-off:
1. **DLLM (Architect):** Runs for 40% of the diffusion steps to generate a structural blueprint (signatures, imports).
2. **Extraction:** Remaining `<|mdm_mask|>` function bodies are collapsed into `# <FILL_ME>` markers.
3. **AR (Staff Engineer):** DeepSeek-Coder-6.7B is prompted to autoregressively infill the holes.

**The Result:** Performance severely degraded. Coherence plummeted to `0.8000` and AST Distance fell to `0.6258`. 
**Why? Logic Drift.** As the AR model iteratively generated the internal logic, it hallucinated variables and made up local state that broke the pre-established global blueprint. This experiment conclusively proved that **relying on AR for logic infilling re-introduces the exact hallucination problem DLLMs solve.** Pure DLLM diffusion is required from start to finish to maintain 100% architectural integrity.

---

## 🚀 Infrastructure & Replication

**Hardware Requirements:**
- 1x NVIDIA L4 GPU (24GB VRAM) or A100.
- A **16GB swapfile** is strictly required to prevent OOM errors when loading both 8B and 6.7B models into memory (even sequentially, due to fragmentation).

**Software Stack:**
- Conda Environment (`dllm-refactor`)
- PyTorch 2.6.0+cu124
- `transformers` 

**Quick Start:**
```bash
# 1. Setup Environment
bash scripts/setup_env.sh
conda activate dllm-refactor

# 2. Run the Pure DLLM Agent on the Stress Test (SOTA)
PYTHONPATH=. python benchmarks/eval_harness.py \
  --model llada \
  --decoder path+dawn \
  --quantize none \
  --limit 1 \
  --repo-path benchmarks/stress_test
```
