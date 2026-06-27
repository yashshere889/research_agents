# Research Agents

A Python research assistant that runs a multi-agent pipeline for scientific
research. It can search literature, formulate a research plan, prepare data,
run experiments, draft a paper, and send each stage through a reviewer team.

The original notebook has been preserved at
`notebooks/research_agent_v3.ipynb`. The reusable implementation lives in the
installable `research_agents` package.

## Pipeline

1. Literature review over arXiv papers
2. Research plan formulation
3. Data preparation in an E2B sandbox
4. Experiment execution
5. Paper drafting and PDF rendering
6. Parallel specialist review after each stage

## Setup

```bash
git clone https://github.com/yashshere889/research_agents.git
cd research_agents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
cp .env.example .env
```

Fill in the credentials you plan to use in `.env`.

Required for Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

Required for Hugging Face models:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_huggingface_token_here
```

Required for E2B-backed data preparation and experiments:

```env
E2B_API_KEY=your_e2b_api_key_here
```

## Run

```bash
research-agents "Your research question here"
```

You can also run it as a module:

```bash
python -m research_agents.main "Your research question here"
```

Resume a saved session:

```bash
research-agents "Your research question here" --resume SESSION_ID
```

Outputs are written to `outputs/` by default. Local vector-store state is written
to `storage/chroma/`.

## Run Gemma 4 On Kaggle

For the Gemma 4 31B instruction model, configure:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_huggingface_token_here
AGENT_MODEL=google/gemma-4-31B-it
REVIEWER_MODEL=google/gemma-4-31B-it
HF_LOAD_IN_4BIT=true
MAX_OUTPUT_TOKENS=1024
MAX_ARXIV_RESULTS=5
MAX_ARXIV_QUERIES=1
```

On Kaggle, enable a GPU accelerator, turn on Internet in the notebook settings,
add your Hugging Face token as a Kaggle secret named `HF_TOKEN`, and run
`notebooks/gemma4_kaggle.ipynb`.

The 31B model is large. If Kaggle runs out of VRAM even with 4-bit loading, use a
smaller Gemma model for testing and switch back when you have a larger GPU.

## Tests

```bash
pytest
```

Unit tests avoid live Gemini, Hugging Face, arXiv, Chroma, and E2B calls.
