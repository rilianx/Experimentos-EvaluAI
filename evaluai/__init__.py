"""Framework para experimentos EvaluAI."""

from .dataset import load_dataset, generate_set
from .prompts import Prompt, generate_prompts
from .optimization import optimize_params, convert_gpt_scores
from .experiments import (
    generate_responses,
    evaluate,
    evaluate_mul,
    read_evals,
    get_files_starting_with,
)
