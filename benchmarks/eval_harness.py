"""
benchmarks/eval_harness.py
Runner script to iterate over the RefactorBench dataset and evaluate the agent.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Fix PYTHONPATH locally to ensure we can run this from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_loader import ModelConfig
from core.refactor_agent import RefactorAgent

def load_refactorbench_dataset(repo_path: str):
    """
    Adapter to load RefactorBench data dynamically from the cloned repositories.
    """
    repos_dir = os.path.join(repo_path, "repositories")
    dataset = []
    
    if os.path.isdir(repos_dir):
        print(f"[INFO] Scanning RefactorBench repositories at: {repos_dir}")
        for task_name in sorted(os.listdir(repos_dir)):
            task_dir = os.path.join(repos_dir, task_name)
            if not os.path.isdir(task_dir): 
                continue
                
            source_files = {}
            # Recursively find small python files to use as refactoring targets
            for root, _, files in os.walk(task_dir):
                for f in files:
                    if f.endswith(".py"):
                        fpath = os.path.join(root, f)
                        # Keep it under 10KB to prevent context window overflow
                        if os.path.getsize(fpath) < 10000:
                            rel_path = os.path.relpath(fpath, task_dir)
                            try:
                                with open(fpath, 'r', encoding='utf-8') as src:
                                    source_files[rel_path] = src.read()
                            except UnicodeDecodeError:
                                pass
            
            # If we found valid python files, create a dataset task
            if source_files:
                # Limit to 3 files max per task to keep generation feasible for 8B models
                limited_files = {k: source_files[k] for k in list(source_files.keys())[:3]}
                dataset.append({
                    "task_id": f"rb_{task_name}",
                    "source_files": limited_files,
                    "instruction": (
                        f"Refactor the codebase in {task_name} to improve structural modularity. "
                        "Convert any monolithic classes into pure functions where applicable, "
                        "and ensure strict adherence to PEP8 naming conventions. "
                        "Maintain all existing cross-file dependencies."
                    )
                })

    if not dataset:
        print(f"[WARN] No valid python tasks found in {repos_dir}. Using dummy test data.")
        return [
            {
                "task_id": "rb_test_001",
                "source_files": {
                    "calculator.py": "class Calc:\n    def add(self, a, b):\n        return a + b\n"
                },
                "instruction": "Refactor the Calc class to functional style standalone functions."
            },
            {
                "task_id": "rb_test_002",
                "source_files": {
                    "string_util.py": "class StringUtil:\n    def to_upper(self, s: str):\n        return s.upper()\n"
                },
                "instruction": "Refactor the StringUtil class to a standalone pure function."
            }
        ]
        
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Run RefactorBench Evaluation")
    parser.add_argument("--model", choices=["llada", "deepseek"], default="deepseek")
    parser.add_argument("--decoder", choices=["none", "dawn", "dawn+streaming", "path+dawn", "hybrid"], default="none")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--repo-path", type=str, default=os.path.expanduser("~/repos/RefactorBench"))
    parser.add_argument("--limit", type=int, default=10, help="Max number of tasks to evaluate")
    parser.add_argument("--k-particles", type=int, default=1, help="Number of SMC particles for POKE-SMC")
    args = parser.parse_args()

    print(f"=== Starting Eval Harness ===")
    print(f"Model: {args.model} | Decoder: {args.decoder}")
    
    dataset = load_refactorbench_dataset(args.repo_path)
    if args.limit:
        dataset = dataset[:args.limit]
        
    cfg = ModelConfig(
        model_type=args.model,
        quantize=args.quantize,
        max_new_tokens=1024,
    )
    agent = RefactorAgent(model_cfg=cfg, decoder=args.decoder, results_dir="results/refactorbench")
    
    success_count = 0
    total_ast_dist = 0.0
    total_coherence = 0.0
    
    for i, item in enumerate(dataset):
        print(f"\n[Task {i+1}/{len(dataset)}] {item['task_id']}")
        result = agent.refactor(
            task_id=item["task_id"],
            source_files=item.get("source_files", {}),
            instruction=item.get("instruction", ""),
            test_dir=item.get("test_dir"),
            k_particles=args.k_particles,
        )
        
        if result.error:
            print(f"  > Error       : {result.error.strip().splitlines()[-1]}")
        elif result.ast_structural_distance is not None:
            total_ast_dist += result.ast_structural_distance
            if result.cross_file_coherence is not None:
                total_coherence += result.cross_file_coherence
            success_count += 1
            print(f"  > AST Distance: {result.ast_structural_distance}")
            print(f"  > Coherence   : {result.cross_file_coherence}")
        else:
            print(f"  > AST Distance: PARSE_ERROR")
            if result.cross_file_coherence is not None:
                print(f"  > Coherence   : {result.cross_file_coherence} (parse error, not counted)")
            
    print("\n=== Benchmark Complete ===")
    if success_count > 0:
        print(f"Average AST Distance: {total_ast_dist / success_count:.4f}")
        print(f"Average Coherence   : {total_coherence / success_count:.4f}")
        print(f"Total Successes     : {success_count}/{len(dataset)}")
    else:
        print("No successful refactorings to aggregate.")

if __name__ == "__main__":
    main()
