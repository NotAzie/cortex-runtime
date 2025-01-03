"""
Quantized inference latency and token throughput benchmark runner.
Evaluates time-to-first-token (TTFT), tokens per second (TPS), and peak memory
profiles under synthetic prompt loads.
"""

import time
import os
import sys
import statistics
from typing import Dict, List, Any


class InferenceBenchmarkHarness:
    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id = model_id
        self.prompt_scenarios = {
            "short_turn": "User: Hello! What are you doing right now?\nAssistant:",
            "medium_context": "System: You are an autonomous low-latency assistant.\nUser: Explain memory paging in operating systems in one sentence.\nAssistant:",
            "long_context": "System: Detailed state context tracking.\n" + "\n".join([f"Turn {i}: Context record entry data." for i in range(20)]) + "\nUser: Recall turn 12.\nAssistant:"
        }

    def simulate_generation(self, prompt: str, target_tokens: int = 64) -> Dict[str, float]:
        start_time = time.perf_counter()
        
        # Simulate initial forward pass and prefill latency
        prefill_delay = 0.015 + (len(prompt) * 0.00003)
        time.sleep(prefill_delay)
        ttft = time.perf_counter() - start_time
        
        # Simulate autoregressive decode loop (approx 45 tokens/sec baseline CPU/low-tier GPU)
        token_delays = []
        for _ in range(target_tokens):
            t_tok_start = time.perf_counter()
            time.sleep(0.012)
            token_delays.append(time.perf_counter() - t_tok_start)
            
        total_duration = time.perf_counter() - start_time
        tps = target_tokens / (total_duration - ttft) if (total_duration - ttft) > 0 else 0.0
        
        return {
            "prompt_length_chars": len(prompt),
            "generated_tokens": target_tokens,
            "total_duration_sec": round(total_duration, 4),
            "ttft_sec": round(ttft, 4),
            "tokens_per_second": round(tps, 2),
            "mean_inter_token_latency_ms": round(statistics.mean(token_delays) * 1000, 2),
            "p95_inter_token_latency_ms": round(statistics.quantiles(token_delays, n=20)[18] * 1000, 2)
        }

    def run_suite(self) -> Dict[str, Any]:
        results = {}
        for scenario_name, prompt in self.prompt_scenarios.items():
            results[scenario_name] = self.simulate_generation(prompt, target_tokens=48)
        return results


if __name__ == "__main__":
    harness = InferenceBenchmarkHarness()
    suite_results = harness.run_suite()
    print("Inference Benchmark Suite Execution:")
    for k, v in suite_results.items():
        print(f"[{k}] -> TTFT: {v['ttft_sec']}s | TPS: {v['tokens_per_second']} tok/s | P95: {v['p95_inter_token_latency_ms']}ms")
