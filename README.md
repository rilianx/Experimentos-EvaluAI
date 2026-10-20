# EvaluAI experiments

Code, prompts, data and results for the paper *EvaluAI: Automatic scoring short-answer constructed-response questions with LLMs*. The version used in the revised manuscript (R2) is tagged `evaluai-R2`.

EvaluAI grades short-answer responses with an LLM (GPT-4o mini). The prompt is built from *mini-prompts* (rubric criteria, knowledge of the expected answer, few-shot examples, analysis, feedback), and the model's 0–10 score is mapped to the four-level rubric (0–3) by a piecewise linear function with two thresholds `a` and `b`.

## Repository layout

| Path | Content |
|---|---|
| `evaluai/` | Python package: dataset loading, prompt building, model queries, score mapping, metrics and analyses |
| `miniprompts/` | Mini-prompts used in the experiments (one folder per component, one `.txt` per variant) |
| `data/` | Standardized dataset (`evaluai_dataset.xlsx`) and the script that builds it (see `data/README.md`) |
| `experiments/` | Experiment suites (JSON) |
| `results/` | Aggregated metrics produced by the suites (no student data) |
| `run_suite.py` | Runs a complete suite: generation, evaluation, paired comparisons and analyses |
| `main.py` | Command-line interface for single experiments (`config.json`) |

## Installation

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
cp .env.example .env              # then set your OpenAI API key
cp config.example.json config.json
```

The API key is read from the environment variable **OPENAI_API_KEY** (or from `.env`).

## Reproducing the paper

| Suite | Content | Results |
|---|---|---|
| `experiments/datastructures_criteria.json` | Data Structures course: all prompt configurations (Tables 3–7), grader agreement (Table 8), sensitivity and calibration (Figures 6–7), strictness and discrimination (Table 10), stability | `results/datastructures_criteria/` |
| `experiments/optimization.json` | Optimization course (Table 9) | `results/optimization/` |
| `experiments/datastructures.json` | Same sequence without the rubric criteria in the prompt (not reported in R2) | `results/datastructures/` |

```bash
python run_suite.py experiments/datastructures_criteria.json --smoke         # quick test
python run_suite.py experiments/datastructures_criteria.json --only base,examples_1G1B
python run_suite.py experiments/datastructures_criteria.json                 # full suite
```

All configurations of a suite are evaluated on the same ten test sets of 100 responses (fixed seeds), so every comparison is paired (Wilcoxon signed-rank test). Model outputs are stored in `runs/<suite>/` and reused on later runs; only new configurations query the model. Aggregated metrics are written to `results/<suite>/`:

| File | Content |
|---|---|
| `summary_mean.csv`, `summary_std.csv`, `summary.md` | Metrics per configuration (mean and standard deviation over the ten test sets) |
| `comparisons.csv` | Paired comparisons between configurations |
| `recalibrated.csv`, `recalibrated_comparisons.csv` | Macro-MSE after recalibrating the thresholds for each configuration, and Spearman correlation |
| `sensitivity.csv` | Macro-MSE for a grid of thresholds `a`, `b` |
| `calibration_*.csv` | Calibration of `a`, `b` with *k* graded responses |
| `stability.csv` | Variability of individual scores across repeated inferences |
| `raw_distribution_*.csv` | Distribution of the model's raw scores per reference level |
| `grader_agreement.csv` | MSE between human graders and the model |

### Running on GitHub Actions

The `Experimentos` workflow runs the suites on GitHub:

1. Add the repository secret `OPENAI_API_KEY` (Settings → Secrets and variables → Actions).
2. Edit `experiments/run-request.json` and push, or start the workflow manually from the Actions tab:

```json
{
  "runs": [{"suite": "experiments/datastructures_criteria.json", "only": ""}],
  "reuse_run_id": "",
  "smoke": false
}
```

`reuse_run_id` downloads the model outputs of a previous run, so that only new configurations are queried. The workflow commits the aggregated metrics to `results/` on the same branch and uploads the full outputs as an artifact.

## Single experiments (`main.py`)

A single experiment is configured in `config.json` (see `config.example.json`):

```bash
python main.py prompts                             # show the prompt(s) to be evaluated
python main.py generate                            # query the model and save responses in Responses/
python main.py evaluate --prefix feedback_normal   # map the scores of Responses/ files starting with the prefix
python main.py read --prefix T_ --normalize        # show the metrics of Evals/ files
python main.py all                                 # generate + evaluate
```

Every subcommand accepts `--config <file>` (default `config.json`). The same functions can be used from Python:

```python
from evaluai import load_dataset, generate_prompts, generate_responses, evaluate_mul, read_evals
```

### Dataset

`load_dataset(path, sheet_name, column_data, query=None)` reads an `.xlsx` file. `column_data` (key `dataset.columns` in `config.json`) maps the required fields to column names:

```json
{"context": "knowledge_detailed", "question": "question", "answer": "answer", "real_eval": "score", "dataset": "subset"}
```

The optional `query` (key `dataset.query`) filters rows, e.g. `"in_paper_290 and reference_source != 'earlier GPT score'"` for the paper dataset.

### Prompts

A prompt is a dictionary whose keys are mini-prompt components (folders of `miniprompts/`) and whose values are variants (`.txt` files in that folder). The wildcard `*` evaluates every variant of a folder.

```json
{
  "examples_basic": "examples_1G1B_basic.txt",
  "context": "context.txt",
  "question": "question.txt",
  "answer": "answer.txt",
  "instructions": {"score": "score_rubric10.txt"}
}
```

`generate_prompts(prompt_data, prompt_folder, visualize=True)` builds the prompt(s). Mini-prompt metadata:

- A first line `#key: description` adds `key` to the output dictionary requested from the model. Without `:` (e.g. `#feedback`), the folder name is used as key.
- In the `score` folder, each line `$criterion` or `$criterion: description` defines a criterion to score. If none is defined, `score` is used.

### Generating responses

`generate_responses(dataset, prompts, repetitions, set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1, balance_set=False, repeat_set=False)` (key `generate`):

- **repetitions**: number of test sets per prompt.
- **set_size**: size of each test set.
- **seed**: seed used to draw the test sets.
- **balance_set**: same number of responses per score level.
- **repeat_set**: use the same test set in every repetition (stability analysis).

### Evaluating responses

`evaluate_mul(filenames, dataset, eval_function, eval_params=None, train_set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1)` (key `evaluate`):

- **eval_function**: mapping from the 0–10 score to the 0–3 scale. The paper uses `map2-simple` (thresholds `a`, `b`).
- **eval_params**: mapping parameters. If `null`, they are fitted by differential evolution on a training set (which queries the model).
- **train_set_size**: size of the training set.

Results are saved in `Evals/` (prefix `T_` when the parameters were fitted). In the suites, each experiment can set `objective` to `mse` or `macro` (Macro-MSE) for the fit.

The original notebook `Framework_EvaluAI.ipynb` is kept for reference; the current code is in `evaluai/`.
