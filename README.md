# Wind-LLM-Agent

A **trustworthy Large Language Model agent for wind-turbine operations and maintenance (O\&M)**.

This repository is the reference implementation for the paper

> Hasan, A., Widyotriatmo, A., Nguyen, D. T. *A Trustworthy Large Language Model Agent for Wind Turbine Operations and Maintenance.*

The agent orchestrates a curated library of **five deterministic tools** through an open-weight LLM (Qwen3 served locally via [Ollama](https://ollama.com)), producing **grounded, auditable, and consistent** answers to operator queries about wind-turbine health and maintenance.

```
operator query  ──►  LLM (Qwen3 / Ollama)  ──►  {T1, T2, T3, T4, T5}  ──►  grounded answer + trace
                          ▲                              │
                          └──────────── tool returns ────┘
```

---

## Key ideas

The paper recasts the wind-farm decision-support problem as a two-stage information-fusion problem:

1. **Feature-level operator** `M_θ` — compresses a sliding window of SCADA into a low-dimensional indicator vector.
2. **Decision-level operator** `F` — fuses indicators, event log, and a natural-language query into a textual response.

`F` is realised by a tool-calling LLM policy `π_LLM` that orchestrates a finite library of deterministic tools. This construction guarantees:

- **Grounding** — every numerical token in the response is the deterministic output of a tool call.
- **Consistency** — those tokens are invariant across repeated runs on the same input (modulo round-off).

These properties are formalised and proved in Propositions 1 and 2 of the paper. They are the architectural reason the agent does not hallucinate component names or fabricate threshold values, which is the dominant failure mode for naked LLM deployments in safety-critical settings.

---

## The five tools

| ID | Name | Input | Output | Source file |
|----|------|-------|--------|-------------|
| `T1` | SCADA Query | turbine id, channels, time window | mean / std / extrema / offline intervals | `src/wind_llm_agent/tools/scada_query.py` |
| `T2` | Anomaly Detector | channel pair, window length, threshold | per-sample anomaly score + crossings | `src/wind_llm_agent/tools/anomaly_detector.py` |
| `T3` | Thermal / Physics Check | channels, time window | list of rule violations | `src/wind_llm_agent/tools/physics_check.py` |
| `T4` | RAG Retriever | natural-language query, top-k | retrieved passages + source ids | `src/wind_llm_agent/tools/rag_retriever.py` |
| `T5` | Maintenance Scheduler | action, horizon | earliest feasible window | `src/wind_llm_agent/tools/maintenance_scheduler.py` |

Each tool is a pure Python function with an explicit JSON-Schema interface (see `src/wind_llm_agent/tools/schemas.py`). The LLM is exposed to tools only through these schemas.

---

## Repository layout

```
wind-llm-agent/
├── README.md                       — this file
├── LICENSE                         — MIT
├── requirements.txt                — Python deps
├── pyproject.toml                  — package metadata
├── configs/
│   └── default.yaml                — model name, thresholds, paths
├── prompts/
│   └── system_prompt.txt           — system prompt σ for the agent
├── src/wind_llm_agent/
│   ├── __init__.py
│   ├── state.py                    — shared state Ξ_k = (Y_k, s_k, H_k)
│   ├── data_loader.py              — load/clean the SCADA CSV
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── schemas.py              — JSON-Schema for all tools
│   │   ├── scada_query.py          — T1
│   │   ├── anomaly_detector.py     — T2
│   │   ├── physics_check.py        — T3
│   │   ├── rag_retriever.py        — T4
│   │   └── maintenance_scheduler.py — T5
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── runtime.py              — tool-call loop (Listing 1 in paper)
│   │   └── registry.py             — name → callable wiring
│   └── baselines/
│       ├── __init__.py
│       ├── threshold_alarm.py      — B1
│       └── isolation_forest.py     — B2
├── scripts/
│   ├── run_agent.py                — single-query CLI
│   ├── make_figures.py             — reproduce Figs 2, 3, 4, 5
│   ├── run_benchmark.py            — Table 4 reproduction
│   └── build_rag_index.py          — build the FAISS index for T4
├── tests/
│   ├── test_scada_query.py
│   ├── test_anomaly_detector.py
│   ├── test_physics_check.py
│   └── test_runtime.py
├── data/
│   └── README.md                   — how to obtain the SCADA dataset
└── docs/
    ├── architecture.md             — extended notes on Fig. 1
    └── adding_a_tool.md            — how to add a sixth tool
```

---

## Installation

Tested on Ubuntu 22.04 / 24.04 and macOS 14, Python 3.10+.

```bash
git clone https://github.com/<your-org>/wind-llm-agent.git
cd wind-llm-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Pull the Qwen3 model via Ollama

The agent loads the model through Ollama's local HTTP API. Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama serve &                 # start the daemon (if not already running)
ollama pull qwen3:14b          # ~9 GB at the default Q4_K_M quantisation
```

You can use a smaller variant (`qwen3:8b`, `qwen3:4b`) for faster iteration on a laptop, or a larger variant (`qwen3:32b`) on a workstation with more VRAM. The model name is read from `configs/default.yaml`.

### Obtain the SCADA dataset

The case-study dataset (a 2 MW Siemens turbine, 30 days at 10-minute resolution, gearbox-bearing fault at sample 1232) is publicly available — see `data/README.md` for the source link and the expected CSV layout. Drop the file as `data/scada.csv` and you are ready to go.

---

## Quick start

### Run a single operator query

```bash
python scripts/run_agent.py \
    --query "Are any turbines showing pre-fault signatures in the next 72 h?" \
    --as-of 2012-11-07T09:00
```

Expected output (abridged — the full trace is logged to `logs/`):

```
[T1] scada_query(channels=[GbxBrgTemp,GenTemp1,GenTemp2,GenSpd], window=72h) → ...
[T2] anomaly_detector(GenSpd, GbxBrgTemp) → crossings since 2012-11-06T06:40
[T3] physics_check(temperatures) → ρ_gbx,gen2 = 0.78 vs 0.97 baseline (rule R3 violated)
[T4] rag_retriever("Siemens 2 MW gearbox bearing fault precursors", k=2) → 2 passages
[T5] maintenance_scheduler(action="borescope inspection", horizon=72h) → 2012-11-08 06:00–14:00

ANSWER:
Turbine T1 shows two pre-fault signatures consistent with an early-stage
gearbox-bearing fault. ...
Confidence: medium-high.
```

### Reproduce the paper figures

```bash
python scripts/make_figures.py --data data/scada.csv --outdir figs/
```

This regenerates `fig_fault_timeseries.pdf`, `fig_power_curve.pdf`, `fig_stationarity.pdf`, and `fig_correlation.pdf`.

### Reproduce Table 4 (benchmark)

```bash
python scripts/run_benchmark.py \
    --data data/scada.csv \
    --queryset configs/eval_queries.json \
    --methods proposed,b1_threshold,b2_iforest,b3_zeroshot
```

---

## How a tool call is wired through the agent

`src/wind_llm_agent/agent/runtime.py` is a direct implementation of equations (7)–(9) in the paper:

```python
def run_agent(messages, tools_schema, tool_impls,
              model="qwen3:14b", max_steps=8):
    for _ in range(max_steps):
        resp = client.chat(model=model, messages=messages,
                           tools=tools_schema, think=False)
        msg = resp["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg["content"]               # terminate → r_k
        for c in calls:                          # conversation update
            name = c["function"]["name"]
            args = c["function"]["arguments"]
            result = tool_impls[name](**args)
            messages.append({"role": "tool",
                             "tool_name": name,
                             "content": json.dumps(result)})
    return BOTTOM                                # failure token ⊥
```

The loop continues until the model emits a final response or the turn budget `J_max = 8` is exhausted, in which case the function returns the abstention token `⊥` rather than fabricating an answer.

---

## Reproducibility notes

- Inference uses temperature `0.2` and top-p `0.9`. The same seed and sampling parameters produce identical *numerical* content across runs; the surrounding prose may vary (this is the consistency property).
- All experiments in the paper used Qwen3-14B at INT4 quantisation on a single 24 GB NVIDIA GPU. Smaller / larger variants work with no code changes — only the `model` field in `configs/default.yaml`.
- The RAG corpus shipped here is a minimal synthetic example (`data/manuals/`) so that the repo is self-contained. Replace it with the operator's real corpus by rebuilding the index with `scripts/build_rag_index.py`.

---

## Citing this work

If you use the code or the framework in academic work, please cite the paper:

```bibtex
@article{hasan2026trustworthy,
  title  = {A Trustworthy Large Language Model Agent for Wind Turbine
            Operations and Maintenance},
  author = {Hasan, Agus and Widyotriatmo, Augie and Nguyen, Dong Trong},
  year   = {2026},
  note   = {Funded under the Erasmus+ GREENSEA project}
}
```

---

## License

MIT — see `LICENSE`. The bundled prompts and synthetic manual examples are released under CC-BY-4.0.

---

## Acknowledgements

This work was supported by the **Erasmus+ GREENSEA** project. The case-study SCADA dataset is reused from publicly available wind-turbine condition-monitoring datasets; see `data/README.md` for full attribution.
