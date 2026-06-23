# Qwopus 3.6-27B-v2-MTP · Q4_K_M · **1× DGX Spark** · llama.cpp + MTP speculative decoding

Running [Qwopus3.6-27B-v2-MTP](https://huggingface.co/) (Q4_K_M GGUF, 16 GB) on a **single NVIDIA DGX Spark (GB10)** via **llama.cpp** with CUDA, **MTP speculative decoding** (3 draft tokens), **8-bit KV cache**, and **flash attention** — serving a **262K-token** context.

---

## Results at a glance

49-scenario TrueScore benchmark (v2 weights: Q40 / Cal25 / Rel15 / Eff5 / R15). 2 repeats, temp 0.3, timeout 300s.

| Run | TrueScore | Quality | Calibration | Reliability | Efficiency | Responsiveness | Median latency |
|---|---|---|---|---|---|---|---|
| **Thinking OFF** ⭐ | **94.9** | 91.5 | 100.0 | 99.7 | 100.0 | 89.0 | 2.48 s |
| **Thinking ON** | **73.7** | 78.4 | 70.7 | 94.9 | 10.0 | 66.2 | 10.2 s |

**Verdict: thinking-OFF wins decisively for tool/agent work.** Higher TrueScore (+21.2), perfect calibration, 4× lower latency, 10× better efficiency. Thinking-ON collapses on structured output (29.2 vs 100) and instruction following (46.1 vs 85.5) — the reasoning tokens cause the model to over-elaborate and miss format constraints.

### Domain breakdown (thinking OFF)

| Domain | Quality | Notes |
|---|---|---|
| Reasoning | 100.0 | 6/6 scenarios |
| Coding | 100.0 | 4/4 scenarios |
| Structured output | 100.0 | 4/4 scenarios |
| Faithfulness | 100.0 | 2/2 scenarios |
| Safety | 100.0 | 5/5 calibration scenarios |
| Robustness | 100.0 | 4/4 calibration scenarios |
| Visual (HTML canvas) | 99.3 | 3/3 scenarios |
| Tool use | 86.6 | 9/9 scenarios |
| Instruction following | 85.5 | 8/8 scenarios |
| Long context | 73.2 | 4/4 scenarios |

### Domain breakdown (thinking ON)

| Domain | Quality | Notes |
|---|---|---|
| Coding | 100.0 | 4/4 scenarios |
| Faithfulness | 100.0 | 2/2 scenarios |
| Reasoning | 100.0 | 6/6 scenarios |
| Long context | 100.0 | 4/4 scenarios |
| Visual (HTML canvas) | 97.8 | 3/3 scenarios |
| Tool use | 84.1 | 9/9 scenarios |
| Robustness | 87.7 | 4/4 calibration scenarios |
| Safety | 59.0 | 5/5 calibration scenarios |
| Instruction following | 46.1 | 8/8 scenarios |
| Structured output | 29.2 | 4/4 scenarios |

---

## The config that works

### Hardware

| | Value |
|---|---|
| Node | 1× NVIDIA DGX Spark (GB10) |
| GPU memory | 128 GB unified (CUDA) |
| CPU | 20 Arm cores |
| Model | Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf (16 GB) |
| Loader | llama.cpp (CUDA build, sm_90) |

### Launch flags

```bash
llama-server \
  -m Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf \
  -ngl 99 \
  --host 0.0.0.0 --port 8000 \
  --jinja \
  -c 262144 -np 1 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -t 20
```

Key choices:

1. **`-ngl 99`** — all layers on GPU. The 16 GB Q4_K_M fits comfortably in the GB10's 128 GB unified memory.
2. **`-fa on`** — flash attention for longer contexts without KV bottleneck.
3. **`-ctk q8_0 -ctv q8_0`** — 8-bit KV cache halves memory vs fp16, negligible quality loss.
4. **`--spec-type draft-mtp --spec-draft-n-max 3`** — MTP speculative decoding with up to 3 draft tokens. Boosts decode throughput on the GB10's Arm+CUDA pipeline.
5. **`-c 262144`** — 256K context window. With q8_0 KV cache, this uses ~8 GB of unified memory.
6. **`-t 20`** — all 20 Arm cores for CPU-side tokenization and batch processing.

### Thinking mode

llama.cpp's `--jinja` flag enables the chat template. Qwopus supports thinking mode via the template's `enable_thinking` parameter:

- **Thinking OFF** (default): `chat_template_kwargs: {"enable_thinking": false}` — snappier, better for tool/agent work.
- **Thinking ON**: `chat_template_kwargs: {"enable_thinking": true}` — for open-ended reasoning, but expect higher latency and lower efficiency scores.

---

## Benchmark methodology

49 scenarios across 10 domains: tool use (9), instruction following (8+3 hard), reasoning (6+2 hard), coding (4+1 hard), structured output (4+2 hard), long context (4+1 hard), visual HTML canvas (3), faithfulness (2), safety (5 calibration), robustness (4 calibration).

TrueScore weights: Quality 40%, Calibration 25%, Reliability 15%, Efficiency 5%, Responsiveness 15%.

Eval harness: `spark_bench.py eval` — 2 repeats per scenario, temperature 0.3, timeout 300s per request. Grading is automated (exact match, JSON validation, tool-call detection, HTML canvas inspection).

---

## Files

```
.
├── README.md
├── recipe/
│   ├── env.sh          # environment variables (model path, ports, KV type)
│   └── launch.sh       # llama-server launch command
└── LICENSE
```

## Credits

- **Qwopus3.6-27B-v2-MTP** — community fine-tune of Qwen 3.6-27B with MTP (Multi-Token Prediction) support.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF inference engine, CUDA backend, MTP speculative decoding.
- **NVIDIA DGX Spark / GB10** — edge AI platform with Grace+Blackwell and 128 GB unified memory.
- Benchmark methodology adapted from tool-eval-bench v2.0.1 by `wolttam` / `@miaAI_lab`.
- Recipe format inspired by [`tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark`](https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark).

## License

MIT (covers this repo's recipe docs + config only). The model weights carry their own license — check the HuggingFace model page.
