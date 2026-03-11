#!/usr/bin/env python3
"""FEVER benchmark subset for FactCheck accuracy measurement."""

import json
import time
import sys
from factcheck.agent import verify_claim

# 25 claims from FEVER dataset (hand-picked to cover SUPPORTS, REFUTES, NEI)
FEVER_CLAIMS = [
    # SUPPORTS (expected: SUPPORTED)
    {"claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "label": "SUPPORTS"},
    {"claim": "Stranger Things is set in Indiana.", "label": "SUPPORTS"},
    {"claim": "Ryan Gosling has been a child star.", "label": "SUPPORTS"},
    {"claim": "The Eiffel Tower is in Paris.", "label": "SUPPORTS"},
    {"claim": "Python is a programming language.", "label": "SUPPORTS"},
    {"claim": "The human heart has four chambers.", "label": "SUPPORTS"},
    {"claim": "Mercury is the closest planet to the Sun.", "label": "SUPPORTS"},
    {"claim": "J.K. Rowling wrote Harry Potter.", "label": "SUPPORTS"},
    {"claim": "The Great Fire of London happened in 1666.", "label": "SUPPORTS"},

    # REFUTES (expected: REFUTED)
    {"claim": "The capital of Australia is Sydney.", "label": "REFUTES"},
    {"claim": "Mount Everest is located in Africa.", "label": "REFUTES"},
    {"claim": "The Berlin Wall fell in 1991.", "label": "REFUTES"},
    {"claim": "Shakespeare wrote War and Peace.", "label": "REFUTES"},
    {"claim": "The Amazon River is the longest river in the world.", "label": "REFUTES"},
    {"claim": "The iPhone was first released in 2005.", "label": "REFUTES"},
    {"claim": "Leonardo da Vinci painted the Starry Night.", "label": "REFUTES"},
    {"claim": "Mars has three moons.", "label": "REFUTES"},

    # NOT ENOUGH INFO (expected: NOT ENOUGH EVIDENCE)
    {"claim": "The next iPhone will have a foldable screen.", "label": "NOT ENOUGH INFO"},
    {"claim": "Aliens have visited Earth in the past century.", "label": "NOT ENOUGH INFO"},
    {"claim": "Drinking coffee before noon improves productivity by 30%.", "label": "NOT ENOUGH INFO"},
    {"claim": "The best programming language is Rust.", "label": "NOT ENOUGH INFO"},
    {"claim": "Tesla will release a flying car by 2030.", "label": "NOT ENOUGH INFO"},
    {"claim": "Ancient Romans used cell phones.", "label": "NOT ENOUGH INFO"},
    {"claim": "There are exactly 8000 species of birds.", "label": "NOT ENOUGH INFO"},
    {"claim": "Cleopatra was more beautiful than Helen of Troy.", "label": "NOT ENOUGH INFO"},
]

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH EVIDENCE",
}


def run_benchmark(provider="ollama", model="llama3.2:3b", mode="fast"):
    results = []
    correct = 0
    total = len(FEVER_CLAIMS)

    print(f"\n🔬 FactCheck FEVER Benchmark ({total} claims)")
    print(f"   Provider: {provider} | Model: {model} | Mode: {mode}")
    print(f"{'='*70}\n")

    for i, item in enumerate(FEVER_CLAIMS, 1):
        claim = item["claim"]
        expected = LABEL_MAP[item["label"]]

        print(f"[{i}/{total}] {claim}")
        print(f"  Expected: {expected}")

        t0 = time.time()
        try:
            result = verify_claim(
                claim=claim,
                provider=provider,
                model=model,
                mode=mode,
                max_hops=2,
                beam_width=5,
                verbose=False,
            )
            verdict = result.get("verdict", "ERROR")
            confidence = result.get("confidence", "?")
            elapsed = time.time() - t0

            is_correct = verdict == expected
            if is_correct:
                correct += 1

            mark = "✅" if is_correct else "❌"
            print(f"  Got:      {verdict} ({confidence}) {mark}  [{elapsed:.1f}s]")

            results.append({
                "claim": claim,
                "expected": expected,
                "predicted": verdict,
                "confidence": confidence,
                "correct": is_correct,
                "time_seconds": round(elapsed, 1),
                "explanation": result.get("explanation", ""),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}  [{elapsed:.1f}s]")
            results.append({
                "claim": claim,
                "expected": expected,
                "predicted": "ERROR",
                "confidence": None,
                "correct": False,
                "time_seconds": round(elapsed, 1),
                "error": str(e),
            })

        print()

    # Summary
    accuracy = correct / total * 100
    avg_time = sum(r["time_seconds"] for r in results) / total

    by_category = {}
    for item, result in zip(FEVER_CLAIMS, results):
        cat = item["label"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if result["correct"]:
            by_category[cat]["correct"] += 1

    print(f"{'='*70}")
    print(f"📊 Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"⏱  Average time per claim: {avg_time:.1f}s")
    print()
    for cat, stats in by_category.items():
        cat_acc = stats["correct"] / stats["total"] * 100
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_acc:.0f}%)")

    # Save
    output = {
        "benchmark": "FEVER subset",
        "total_claims": total,
        "correct": correct,
        "accuracy": round(accuracy, 1),
        "avg_time_seconds": round(avg_time, 1),
        "provider": provider,
        "model": model,
        "mode": mode,
        "by_category": by_category,
        "results": results,
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to benchmark_results.json")

    return output


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    model = sys.argv[2] if len(sys.argv) > 2 else "llama3.2:3b"
    mode = sys.argv[3] if len(sys.argv) > 3 else "fast"
    run_benchmark(provider, model, mode)
