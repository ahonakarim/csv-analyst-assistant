# CSV Analyst Assistant

A Python + Streamlit app that helps you analyze CSV files by translating user questions into **structured intents** and running **validated Pandas operations** to produce trustworthy tables, summaries, and charts.

## Overview

- **What it does**: Upload a CSV, ask natural-language questions, get computed answers + charts.
- **How it works**: LLM → structured intent → validation/safety checks → Pandas execution → formatted results.
- **Why structured**: Reduces hallucinations and makes evaluation repeatable.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

- Copy `.env.example` to `.env`
- Fill in provider keys (if using an LLM API)

## Running the app

```bash
streamlit run app/main.py
```

## Frontend (MVP)

The Streamlit frontend lets you:

- upload a CSV file
- preview the dataset and view basic dataset stats/columns
- ask a natural-language question (or use example question buttons)
- view the interpreted structured query
- view returned outputs (message, scalar value, table, chart when available)

Supported query types are limited to:

- `summary_stat`
- `grouped_aggregation`
- `filtered_aggregation`
- `time_series`

## Evaluation

This repo includes scripts and data scaffolding for benchmarking the router/intent parsing and end-to-end correctness.

- **Benchmarks input**: `data/benchmarks/benchmark_questions.csv`
- **Run router-focused eval**: `scripts/run_router_eval.py`
- **Run end-to-end eval**: `scripts/run_end_to_end_eval.py`
- **Outputs**:
  - `outputs/eval_runs/` (structured run artifacts)
  - `outputs/figures/` (plots)
  - `outputs/logs/` (logs)

## Repository structure

```
csv-analyst-assistant/
  app/                      # Streamlit app + core logic
  data/                     # raw/processed/benchmarks
  docs/                     # scope and evaluation docs
  notebooks/                # exploration / benchmark design
  outputs/                  # eval artifacts, figures, logs
  scripts/                  # runnable evaluation utilities
  tests/                    # unit tests
  README.md
  requirements.txt
  .gitignore
  .env.example
```
# ise547project
