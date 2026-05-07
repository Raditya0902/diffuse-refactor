# Project: Non-Linear Code Refactoring via DLLMs

## Quick Commands
- **SSH into VM:** `gcloud compute ssh dllm-refactor-vm --zone=us-east1-b --project=dllm-refactor-2026`
- **Copy Files to VM:** `gcloud compute scp --recurse . dllm-refactor-vm:~/dllm-refactor --zone=us-east1-b`
- **Activate Conda:** `conda activate dllm-refactor`
- **Run Agent (Baseline):** `PYTHONPATH=. python core/refactor_agent.py --model deepseek --decoder none`

## Active Directives (Hooks & Habits)
> [!IMPORTANT]
> **Build Checking:** After modifying ANY Python file, run a quick syntax check (`python -m py_compile <file>`) or test (`pytest`) BEFORE considering the task complete. No Mess Left Behind.

> [!TIP]  
> **Skills Activation:** Before writing or modifying core logic, review `.gemini/skills/ml-python-guidelines.md`.

## Project Context
Building a high-performance refactoring agent to prove that LLaDA 1.5 (DLLM) maintains superior global structural integrity over DeepSeek-Coder-6.7B (AR).
- **Phase 0 & 1 (DONE):** VM provisioned, Conda/CUDA/PyTorch env mapped, Core decoders written.
- **Phase 2 (DONE):** Evaluate via RefactorBench. (Avg AST Dist: 0.9356 | Avg Coherence: 1.0)
- **Phase 3 (DONE):** Path-Guided Unmasking via AST graph (Topological sorting). High-density stress test passed with perfect coherence.

**Status:** Thesis successfully proven. DLLM global structural integrity validated over AR baselines.
