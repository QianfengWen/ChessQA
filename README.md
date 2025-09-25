# ChessQA: Evaluating Large Language Models for Chess Understanding

ChessQA is a comprehensive, dynamic benchmark that evaluates LLM chess understanding across five ascending levels of abstraction — from basic rules to high‑level semantic reasoning — with objective ground truth and reproducible pipelines for dataset construction, inference, and analysis.

**Abstract**
- Chess provides an ideal testbed for evaluating the reasoning, modeling, and abstraction capabilities of large language models (LLMs), as it has well-defined structure and objective ground truth while admitting a wide spectrum of skill levels.
- However, existing evaluations of LLM chess ability are ad hoc and narrow in scope, making it difficult to accurately measure chess understanding and how it varies with scale, post-training methodologies, or architecture choices.
- We present ChessQA, a comprehensive benchmark that assesses LLM chess understanding across five task categories (Structural, Motifs, Short Tactics, Position Judgment, and Semantic), corresponding to the ascending abstractions that players master as they accumulate chess knowledge: from understanding rules and motifs, to calculating tactics, evaluating positions, and describing high‑level concepts.
- ChessQA captures a more holistic picture of chess ability than prior move‑quality evaluations and offers a controlled, consistent setting for diagnosis and comparison. It is inherently dynamic, with prompts, answer keys, and construction scripts that can evolve as models improve.
- Evaluating a range of contemporary LLMs, we find persistent weaknesses across all five categories and provide results and error analyses by category. We release code, refreshed datasets, and a public leaderboard to support further research.

## What’s Inside

- Five categories with objective answer keys and robust extraction
  - Structural: piece arrangement, legal moves (piece/all), check detection and check‑in‑1, capture/control/protect squares, and state tracking (FEN after UCI sequences)
  - Motifs: pin, fork, skewer, battery, discovered check, double check
  - Short Tactics: best‑move puzzles by rating buckets (beginner→expert) and by theme (dozens of tactical themes)
  - Position Judgment: centipawn evaluation selection across bands (neutral/advantage/winning/…)
  - Semantic: multiple‑choice commentary understanding with several distractor strategies (keyword, piece+stage, semantic embedding, easy random)
- OpenRouter evaluation runner with parallelism, resume, cost/tokens tracking, per‑category and per‑task‑type stats
- Dynamic dataset builders that regenerate as the underlying sources improve

## Repository Layout

- `code/dataset`: dataset generation scripts for each category
- `code/eval`: OpenRouter inference runner and result browser
- `data/raw`: source data (Lichess puzzle/eval dumps, PGNs)
- `benchmark`: generated benchmark JSONL files (one per category)
- `results`: per‑model outputs (`*.jsonl`, `*_pretty.json`, `*_stats.json`)

## Install

- Python 3.8+
- `pip install -r requirements.txt`
- Optional for semantic MCQ embeddings: `pip install sentence-transformers faiss-cpu`

API keys
- Set `OPENROUTER_API_KEY` in your environment, or put it in `api_keys.json` (see `setup.sh`).

## Data

Place the following under `data/raw/` (paths match defaults in scripts):
- `lichess_db_puzzle.csv` — Lichess puzzle dump
- `lichess_db_eval.jsonl.zst` — Engine evaluations (for Position Judgment)
- `lichess_db_broadcast_2025-04.pgn` — PGN stream (for Structural state tracking)
- Optional: `chessbase.pgn` / `filtered_chessbase.pgn` — additional PGNs

## Create Benchmark Datasets

Structural
```bash
python code/dataset/01_structural.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --pgn_path data/raw/lichess_db_broadcast_2025-04.pgn \
  --output_root data/benchmark --N_sample 100
```

Motifs
```bash
python code/dataset/02_motif.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --output_root data/benchmark --N_sample 100
```

Short Tactics
```bash
python code/dataset/03_tactic.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --all_themes_path data/info/all_themes_to_include.json \
  --output_root data/benchmark --N_sample_rating 100 --N_sample_theme 25
```

Position Judgment
```bash
python code/dataset/04_judgement.py \
  --data_path data/raw/lichess_db_eval.jsonl.zst \
  --output_root data/benchmark --tasks_per_category 100 --max_evaluations 10000
```

Semantic (MCQ from commentary; requires sentence-transformers)
```bash
python code/dataset/05_semantic.py \
  --input data/mid/comment_dataset.final.json \
  --output_root data/benchmark --N_sample_mcq 100
```

Note: comment cleaning and judging helpers for producing `data/mid/comment_dataset.final.json` live in `code/dataset/05_2_comment_cleaning.py` and `code/dataset/05_3_comment_judging.py` (optional, uses vLLM for offline filtering).

## Run Inference (OpenRouter)

Basic run
```bash
OPENROUTER_API_KEY=... python code/eval/run_openrouter.py \
  --dataset-root benchmark \
  --model anthropic/claude-3.5-haiku \
  --output-dir results --workers 256
```

Options
- Limit total tasks: `--max-tasks 800`
- Uniform sampling per task type: `--N-samples-per-task 50`
- Add auto‑generated context (piece arrangement + legal moves): `--add-context`
- Enable “thinking” for models that support it: `--enable-thinking`
- Re‑evaluate an existing JSONL without calling APIs: `--eval-only`

Outputs
- `results/<model>.jsonl`: per‑task records with prompts/responses and extraction
- `results/<model>_pretty.json`: summarized JSON for quick reading
- `results/<model>_stats.json`: accuracy, per‑category breakdowns, error shares, cost and tokens

## License

MIT License. See `LICENSE` for details.
