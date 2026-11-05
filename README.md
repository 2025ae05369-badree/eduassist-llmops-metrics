# EduAssist AI — LLMOps: API Cost, Latency, and Prompt Version Management

MLOps Experiential Learning Assignment 2. An academic question-answering pipeline that calls an LLM, measures
latency, computes token-based cost, and versions / evaluates / promotes prompts in the **MLflow Prompt Registry**.

## Why a local Hugging Face model instead of a paid API

A paid hosted API (e.g., OpenAI) requires an account with billing set up, which was not available. Section 7 of the
brief allows the alternative: *"use a locally hosted open-source model and clearly document the alternative; estimate
token counts and costs using the model tokenizer and published hosting costs."* Therefore:

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` — open weights, ungated (no account/token), Qwen Research License (academic use) |
| Model revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| "API version" | `transformers 5.22.1`, `torch 2.15.0`, `mlflow 3.17.2` |
| Decoding | greedy (`do_sample=False`, ≙ temperature 0), `max_new_tokens=512` |
| Hardware used | Apple M1 Pro, 16 GB RAM, macOS (arm64), PyTorch `mps` backend, float16 |
| Token counting | model tokenizer (`input_ids` length after chat template / newly generated tokens) |
| Pricing source | [Hugging Face Inference Endpoints pricing](https://huggingface.co/docs/inference-endpoints/en/pricing) — AWS `nvidia-t4` ×1 at **$0.50/hour** |
| Pricing date / currency | **2026-11-02 / USD** |

No per-token price is published for this model (its only Hub inference provider, Featherless AI, is
subscription-based), so per-token prices are **derived** in the notebook from the hourly rate and the measured
throughput: `price_per_1M = hourly_usd / (tokens_per_sec × 3600) × 1,000,000` (prefill throughput for input tokens,
decode throughput for output tokens). Costs are estimates, not bills.

`Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) was tried first, but it wrapped every v3 answer in ```` ```json ```` code
fences (strict `valid_json` = 0.0), so the 3B model is used. `MODEL_NAME` is one constant in the notebook if you want
to swap models.

## Project layout

```
eduassist-llmops-metrics/
├── eduassist_llmops.ipynb        # main notebook (Tasks 1–3), executed with outputs
├── eduassist_llmops.html         # exported notebook
├── report.pdf                    # 500–700 word report
├── requirements.txt
├── README.md
├── .env.example                  # variable names only
├── outputs/
│   ├── latency_tokens_cost.csv          # Task 1 + Task 2 per-request table
│   └── prompt_evaluation_results.csv    # Task 3 comparison table
└── screenshots/                  # MLflow UI screenshots (see "Screenshots" below)
```

## Requirements

* **Python 3.11+** (developed and run with Python 3.12.6)
* ~6 GB free disk for the model weights, ~8 GB free RAM. Apple Silicon (MPS), an NVIDIA GPU (CUDA), or CPU
  (CPU works but is much slower).
* Libraries (`requirements.txt`): `mlflow>=3.17,<4`, `transformers>=4.56`, `torch>=2.3`, `accelerate`,
  `huggingface_hub`, `pandas`, `numpy`, `python-dotenv`, `jupyter`, `ipykernel`, `nbconvert`.

Exact versions used for the recorded run:

| Package | Version | Package | Version |
|---|---|---|---|
| mlflow | 3.17.2 | pandas | 3.1.0 |
| transformers | 5.22.1 | numpy | 2.5.4 |
| torch | 2.15.0 | python-dotenv | 1.2.4 |
| accelerate | 1.15.0 | jupyter | 1.1.1 |
| huggingface_hub | 2.4.0 | ipykernel | 7.4.0 |
| nbconvert | 7.17.2 | nbformat | 5.11.1 |

## Environment variables (all optional)

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token — only for gated models or higher download rate limits. Never hard-code it. | unset (not needed) |
| `MLFLOW_TRACKING_URI` | MLflow tracking backend (a database backend is required for the Prompt Registry) | `sqlite:///mlflow.db` |

To set them, copy `.env.example` to `.env` and fill in values; the notebook loads `.env` with `python-dotenv` and
only prints whether each variable is set. `.env`, `mlflow.db`, `mlruns/`, and `mlartifacts/` are git-ignored.

## Installation

```bash
cd eduassist-llmops-metrics
python3 -m venv .venv            # Python 3.11+
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The **first run downloads the model (~6 GB)** to the Hugging Face cache (`~/.cache/huggingface/hub`).

## Reproducing the experiment

1. Create and activate the virtual environment and install requirements (above).
2. For a clean registry, delete any previous `mlflow.db`, `mlruns/`, and `outputs/`, and restart the Jupyter kernel
   so only one copy of the model is in memory (a second copy on a 16 GB machine causes swapping, which distorts the
   measured throughput and therefore the derived prices).
3. Run the notebook top-to-bottom — either open `eduassist_llmops.ipynb` in Jupyter/VS Code and *Run All*, or:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=3600 eduassist_llmops.ipynb
   ```

   On an M1 Pro this takes ~6 minutes after the model is cached. It runs a warm-up + throughput calibration,
   Task 1 (5 calls), Task 2 (cost tables), and Task 3 (3 MLflow runs × 5 calls), writes both CSVs to `outputs/`,
   and sets the `production` alias.
   Re-running against an existing `mlflow.db` reuses the registered prompt versions (no duplicates) and adds a new
   set of runs.
4. Export the notebook: `jupyter nbconvert --to html eduassist_llmops.ipynb`
5. Start the MLflow UI and open <http://127.0.0.1:5000>:

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```

6. Outputs: `outputs/latency_tokens_cost.csv`, `outputs/prompt_evaluation_results.csv`; MLflow run artifacts
   `responses/prompt_v{1,2,3}_responses.json` hold every raw response.

> The markdown interpretation cells quote the numbers of the recorded run (2026-11-05). Greedy decoding keeps token
> counts and responses reproducible, but latencies (and therefore the derived prices) vary slightly between runs.

### Screenshots (`screenshots/`)

Captured from the MLflow 3.17.2 UI for the recorded run.

| Assignment deliverable | File(s) | What it shows |
|---|---|---|
| Experiment runs and metrics | `mlflow_experiment_runs.png` | *eduassist_llmops_assignment → Evaluation runs*: runs `prompt_v1..3` with `avg_latency_seconds`, `avg_input_tokens`, `avg_output_tokens`, `avg_cost_per_request`, `valid_json` columns |
| Metrics comparison | `mlflow_metrics_comparison_1.png`, `mlflow_metrics_comparison_2.png` | The three runs compared as charts: latency, tokens, cost (part 1); tokens, cost, `valid_json`, error and truncation rates (part 2) |
| Prompt Registry v1, v2, v3 | `prompt_registry_versions_1.png`, `prompt_registry_versions_2.png`, `prompt_registry_versions_3.png` | *Prompts → eduassist-qa*: each version's template, commit message, tags and model config, with the full version list |
| Prompt diff | `prompt_diff_v2_v3.png` | *Compare* view of version 3 against version 2 (highlighted diff) |
| Production alias | `production_alias.png` | Version 3 carrying the `@production` alias |

To recapture them, start the UI (step 5) and avoid panels that show local absolute paths (e.g., *Artifact location*),
API tokens, or account details.

## Results summary (recorded run)

| Prompt | Avg latency (s) | Avg input tokens | Avg output tokens | Avg cost (USD) | valid_json | Recommended |
|---|---|---|---|---|---|---|
| v1 | 18.869 | 51.2 | 413.8 | 0.00258240 | 0.0 | No |
| v2 | 12.063 | 73.2 | 270.2 | 0.00169560 | 0.0 | No |
| v3 | 5.638 | 107.2 | 123.4 | 0.00079175 | 1.0 | **Yes → `production`** |

Task 1 (v1 baseline): mean latency 18.44 s; projected monthly cost for 10,000 requests: $25.82 (v1) vs ≈ $7.92 (v3).
Derived prices: $0.2354 per 1M input and $6.2116 per 1M output tokens (measured prefill 589.9 tokens/s, decode
22.4 tokens/s).
