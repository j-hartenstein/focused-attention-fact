# Data

Processed data is gitignored (generated locally or on Modal).

## Prerequisites

- **HuggingFace token** with access to `meta-llama/Llama-3.2-3B-Instruct` (gated model).
  Request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct, then:
  ```bash
  uv run huggingface-cli login
  ```
- **OpenAI API key** for the `--llm-judge` flag (GPT-4o-mini).
  Add to `.env` at the project root:
  ```
  OPENAI_API_KEY=sk-...
  ```

## Sources

| Task | Source | Script |
|------|--------|--------|
| Jailbreaks | [Mechanistic-Anomaly-Detection/llama3-jailbreaks](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-jailbreaks) | `data/scripts/prepare_jailbreaks.py` |
| Sycophancy | [raybears/cot-transparency](https://github.com/raybears/cot-transparency/tree/main/dataset_dumps) + ARC-Challenge fallback | `data/scripts/prepare_sycophancy.py` |

## Prepare

Download the raybears files (control + bct, non-CoT and CoT variants):
```bash
mkdir -p data/raw/cot-transparency/dataset_dumps/train_seed_42 \
         data/raw/cot-transparency/dataset_dumps/control_seed_42

BASE=https://raw.githubusercontent.com/raybears/cot-transparency/main/dataset_dumps
curl -L -o data/raw/cot-transparency/dataset_dumps/train_seed_42/bct_non_cot.jsonl   $BASE/train_seed_42/bct_non_cot.jsonl
curl -L -o data/raw/cot-transparency/dataset_dumps/train_seed_42/bct_cot.jsonl       $BASE/train_seed_42/bct_cot.jsonl
curl -L -o data/raw/cot-transparency/dataset_dumps/control_seed_42/control_non_cot.jsonl $BASE/control_seed_42/control_non_cot.jsonl
curl -L -o data/raw/cot-transparency/dataset_dumps/control_seed_42/control_cot.jsonl     $BASE/control_seed_42/control_cot.jsonl
```

Then prepare processed pairs:
```bash
# Regex annotation (no OpenAI key needed):
uv run python data/scripts/prepare_jailbreaks.py --n-samples 2000
uv run python data/scripts/prepare_sycophancy.py --n-samples 2000

# With LLM judge for accurate direction annotation (~$0.10):
uv run python data/scripts/prepare_sycophancy.py --n-samples 2000 --llm-judge
```

Or on Modal (see `modal_app/run_experiment.py`):
```bash
modal run modal_app/run_experiment.py::prepare_data
```
