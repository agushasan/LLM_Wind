# LLM Fusion Agent for Industrial Anomaly Detection

Reference implementation of the paper **"Large Language Models as Information
Fusion Agents for Anomaly Detection in Safety-Critical Systems"**
(Hasan, Maione, Sukarevicius, Osen, 2026).

A two-level information-fusion pipeline that combines a Transformer
Autoencoder (feature-level fusion) with a tool-grounded LLM (decision-level
fusion). The autoencoder summarises the joint behaviour of $d$ sensor
channels through a reconstruction residual; the LLM, restricted to a small
library of deterministic tools, fuses those outputs with the running event
history and the operator's natural-language query to produce a single
grounded recommendation.

```
                      +---------------------- FEATURE LEVEL ---------------------+
                      |  sensors  ->  window  ->  Transformer AE  ->  indicators  |
                      +-----------------------------|---------------------------+
                                                    | writes
                                                    v
                      +---------------------- SHARED STATE -----------------------+
                      |   S_t = ( X_t , s_t , H_t )    (window | indicators | log) |
                      +-----------------------------|---------------------------+
                                                    | reads (via tools)
                                                    v
                      +---------------------- DECISION LEVEL ---------------------+
                      |  operator <-> LLM policy  <->  tool library  T_1 ... T_6  |
                      +----------------------------------------------------------+
```

This repository contains a clean, self-contained reference implementation
that lets you reproduce the architecture end-to-end on synthetic data, and
that is straightforward to retarget to a different asset.

---

## What is implemented

| Paper section | Module |
| --- | --- |
| §3.1 Transformer Autoencoder $\mathcal{M}_\theta$ | `src/model/transformer_autoencoder.py` |
| §3.1 Standardisation and sliding-window dataset (Eqs. 1, 5, 6) | `src/model/data.py` |
| §3.1.4 Anomaly score, severity, per-variable attribution (Eqs. 17–20) | `src/model/scoring.py` |
| §3.1.7 Prognostic extrapolation (Eq. 22) | `src/model/prognostic.py` |
| §2 Shared state $\mathbf{S}_t$ | `src/fusion/shared_state.py` |
| Algorithm 1 (online runner) | `src/fusion/runner.py` |
| §3.2.2 Tool library $\mathcal{T} = \{T_1,\ldots,T_6\}$ (Table 2) | `src/tools/library.py` |
| §3.2.3–§3.2.4 LLM agent + bounded multi-turn protocol | `src/fusion/agent.py` |
| §4.3 Fault-injection scenarios | `src/fault_injection/profiles.py` |
| §5.2 Gradio interface | `src/ui/app.py` |

The default hyperparameters match Table 1 of the paper exactly
(`d = 16`, `L = 120`, `d_m = 64`, `H = 4`, `N_e = N_d = 2`, FFN = 256,
dropout = 0.1, AdamW with $\eta = 10^{-4}$ and weight decay $10^{-5}$,
batch size 512, 50 epochs).

---

## Installation

The implementation has been tested with Python 3.10+.

```bash
git clone https://github.com/ModeS7/LLM-maintenance.git
cd LLM-maintenance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the LLM agent you additionally need Ollama and the Qwen3 8B model:

```bash
# https://ollama.com/download — install Ollama for your OS, then:
ollama pull qwen3:8b
ollama serve   # leaves the local API running on http://localhost:11434
```

The autoencoder and tool library do not require Ollama; you can use them
on their own for purely numerical experiments.

---

## Quickstart

The case-study vessel data are not redistributable. The repo therefore
ships a synthetic generator that mirrors the marginal distributions
described in §4.1 (bimodal propulsion power, speed concentrated near
10–12 knots, thrusters near zero, electrical bus loads with a slow
random walk), so the pipeline runs end-to-end out of the box.

```bash
# 1. Generate a small synthetic dataset (~3.5 days at 5 s sampling)
python examples/generate_synthetic_dataset.py --out data/synthetic.npy

# 2. Train the Transformer Autoencoder on the nominal portion
python scripts/train.py \
    --data data/synthetic.npy \
    --out runs/synth1

# 3. Run a fault scenario end-to-end
python scripts/run_scenario.py \
    --run-dir runs/synth1 \
    --data data/synthetic.npy \
    --channel-names data/synthetic.channels.txt \
    --scenario slow_drift \
    --t-start 5000 --t-end 30000
```

Add `--ask` to step 3 to also send a Phase-3 query to the LLM agent
(requires Ollama running with Qwen3 8B).

To explore the four scenarios interactively in a browser:

```bash
# copy the channel-names file next to the run artefacts
cp data/synthetic.channels.txt runs/synth1/channels.txt

python -m src.ui.app --run-dir runs/synth1 --data data/synthetic.npy
```

---

## Reproducing the case study

The numbers reported in Section 5 of the paper come from ninety-one days
of IAS recordings collected aboard an offshore construction vessel. The
dataset is not redistributable due to commercial restrictions, but the
pipeline reproduces with any compatible NumPy array of shape `(N, 16)`,
provided the sixteen channel names follow the convention in
`src/tools/library.py::VARIABLE_GROUPS`.

To reproduce against your own data:

1. Export your IAS recording to a `(N, 16)` `float32` `.npy` array,
   sampled at 5 s, in the channel order:

   ```
   Bus1_Load, Bus2_Load, Bus1_Avail_Load, Bus2_Avail_Load,
   Main_Prop_PS_ME1_Power, Main_Prop_PS_ME2_Power,
   Main_Prop_SB_ME1_Power, Main_Prop_SB_ME2_Power,
   BowThr3_Power, StnThr1_Power, StnThr2_Power,
   Speed, Draft, Latitude, Longitude, Heading
   ```

2. Write the same channel names to `data/your_dataset.channels.txt`
   (one name per line).

3. Run `scripts/train.py --data data/your_dataset.npy --out runs/vessel`.

4. Then run any of the four scenarios with `scripts/run_scenario.py`.

The four fault scenarios are wired exactly as described in §4.3:

| Scenario | Target channels | Tests |
| --- | --- | --- |
| `slow_drift` | `Bus1_Load` | trend extrapolation + correct attribution |
| `load_imbalance` | `Bus1_Load` / `Bus2_Load` | feature-level fusion across correlated channels |
| `temporary_reduction` | `Bus1_Avail_Load` | graceful recovery |
| `short_spikes` | `Bus1_Load` | false-alarm suppression by the 50-sample smoother |

---

## Tests

The non-LLM building blocks ship with `pytest` unit tests:

```bash
pytest tests/
```

These cover: standardisation round-trip, window construction, anomaly
scoring, severity tiers, per-variable attribution, the linear prognostic
fit, the autoencoder forward pass, and each of the four fault profiles.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── default.json                 # hyperparameters from Table 1
├── examples/
│   └── generate_synthetic_dataset.py
├── scripts/
│   ├── train.py                     # §5.1 training loop
│   └── run_scenario.py              # apply a fault + (optionally) ask the LLM
├── src/
│   ├── model/
│   │   ├── transformer_autoencoder.py   # §3.1 — MHA, encoder, decoder, reconstruction
│   │   ├── data.py                       # §3.1 — standardisation + sliding-window dataset
│   │   ├── scoring.py                    # §3.1.4 — error, score, severity, attribution
│   │   └── prognostic.py                 # §3.1.7 — online linear trend
│   ├── fault_injection/
│   │   └── profiles.py                   # §4.3 — four scenarios
│   ├── fusion/
│   │   ├── shared_state.py               # the S_t bridge
│   │   ├── runner.py                     # Algorithm 1
│   │   └── agent.py                      # §3.2.3–4 — bounded multi-turn LLM loop
│   ├── tools/
│   │   └── library.py                    # §3.2.2, Table 2 — six deterministic tools
│   └── ui/
│       └── app.py                        # §5.2 — minimal Gradio interface
└── tests/
    └── test_core.py
```

---

## Architectural guarantees

The construction in §3.2 secures four properties **by design**, which the
implementation here preserves:

* **Grounding.** Every numerical token in the LLM's reply is copied
  verbatim from a tool return value. The tool dispatcher
  (`src/fusion/agent.py`) routes every model-emitted tool call to a
  deterministic Python function over the shared state; the model never
  sees raw arrays directly.
* **Consistency.** Tool outputs are deterministic to floating-point
  precision; only the surface prose around them is sampled.
* **Adaptability.** Changes to the data schema or to the autoencoder
  touch only the tool implementations and the data loader. The LLM
  requires no retraining.
* **Graceful degradation.** A tool failure surfaces as a structured
  error inside the conversation rather than as a silent hallucination.
  If the LLM is unavailable, the Gradio dashboard still displays the
  underlying `s_t` and `H_t` directly.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Hasan2026LLMFusion,
  title   = {Large Language Models as Information Fusion Agents
             for Anomaly Detection in Safety-Critical Systems},
  author  = {Hasan, Agus and Maione, Francesco and Sukarevicius, Modestas
             and Osen, Ottar},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
