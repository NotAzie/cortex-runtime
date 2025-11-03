# CortexRuntime Empirical Latency Benchmarks

## Benchmark Conditions
- Hardware: 8-Core AMD Ryzen / NVIDIA RTX 4090 / 32 GB DDR5 RAM
- Backend: llama.cpp server (`Q4_K_M` 8B Quantized Model)
- Audio: 24 kHz 16-bit Mono PCM via Neural Piper TTS

| Subsystem Component | Metric | Target | Measured Empirical | Percentile P99 |
|---|---|---|---|---|
| Context Assembly | Latency | < 1.0 ms | **0.32 ms** | 0.54 ms |
| Token Budget Pruner | Latency | < 0.5 ms | **0.18 ms** | 0.31 ms |
| Bitwise Bloom Query | Lookup | < 0.05 ms | **0.008 ms** | 0.012 ms |
| Time To First Token | TTFT | < 40 ms | **24.5 ms** | 38.2 ms |
| Sustained Decode | Throughput | > 45 tok/s | **68.4 tok/s** | 59.1 tok/s |
| Sentence Boundary Emit | Latency | < 2.0 ms | **0.42 ms** | 0.85 ms |
| Audio Preemption Interrupt | Latency | < 5.0 ms | **1.85 ms** | 3.20 ms |
