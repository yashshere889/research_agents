# Multi-Agent LLM System for Scientific Research

This package implements an autonomous research pipeline:

1. Literature review over arXiv papers
2. Research plan formulation
3. Data preparation in an E2B sandbox
4. Experiment execution
5. Paper drafting and PDF rendering
6. Parallel specialist review after each stage

## Setup

```bash
cd /Users/yash/Desktop/Summer\ Thesis
python -m venv .venv
source .venv/bin/activate
pip install -r research_agents/requirements.txt
cp research_agents/.env.example .env
```

Fill in `GEMINI_API_KEY` and `E2B_API_KEY`, then run:

```bash
python -m research_agents.main "Your research question here"
```

## Run with Gemma 4 on Kaggle

This project can also use Hugging Face Transformers instead of Gemini. For the
Gemma 4 31B instruction model, use:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=hf_...
AGENT_MODEL=google/gemma-4-31B-it
REVIEWER_MODEL=google/gemma-4-31B-it
HF_LOAD_IN_4BIT=true
MAX_OUTPUT_TOKENS=1024
MAX_ARXIV_RESULTS=5
MAX_ARXIV_QUERIES=1
```

On Kaggle, enable a GPU accelerator, turn on Internet in the notebook settings,
add your Hugging Face token as a Kaggle secret named `HF_TOKEN`, and run
`research_agents/notebooks/gemma4_kaggle.ipynb`. The notebook checks which
packages Kaggle already has, installs only missing packages, writes a Kaggle-safe
`.env`, verifies Gemma 4 generation, then runs the research pipeline.

The 31B model is large. If Kaggle runs out of VRAM even with 4-bit loading, use a
smaller Gemma 4 model for testing and switch back to `google/gemma-4-31B-it`
when you have a larger GPU.

If your Google AI project has no quota for `gemini-2.5-pro`, set this in `.env`:

```env
AGENT_MODEL=gemini-2.5-flash
```

If arXiv returns HTTP 429, wait a few minutes and reduce literature retrieval load:

```env
MAX_ARXIV_RESULTS=10
MAX_ARXIV_QUERIES=1
ARXIV_DELAY_SECONDS=15
```

## Tests

```bash
pytest
```

Unit tests avoid live Gemini/Hugging Face, arXiv, Chroma, and E2B calls.
