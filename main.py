"""CLI para ejecutar experimentos EvaluAI.

Ejemplos:
    python main.py generate --config config.json
    python main.py evaluate --config config.json --prefix feedback_normal
    python main.py read --prefix T_ --normalize
    python main.py all --config config.json
"""
import argparse
import json
import os
import subprocess
import sys

from evaluai import (load_dataset, generate_prompts, generate_responses, evaluate_mul,
                     read_evals, get_files_starting_with)

GPT_EVALUATOR_REPO = "https://github.com/rilianx/GPTEvaluator"


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_env(path=".env"):
    # Carga variables desde un archivo .env simple (CLAVE=valor), sin sobrescribir las existentes
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


# Clona GPTEvaluator si la carpeta de miniprompts por defecto no existe
def ensure_prompt_folder(prompt_folder):
    if os.path.isdir(prompt_folder): return
    if prompt_folder.startswith("GPTEvaluator/") and not os.path.exists("GPTEvaluator"):
        print(f"Clonando {GPT_EVALUATOR_REPO} para obtener los miniprompts...")
        subprocess.run(["git", "clone", "--depth", "1", GPT_EVALUATOR_REPO, "GPTEvaluator"], check=True)
    if not os.path.isdir(prompt_folder):
        sys.exit(f"Error: La carpeta de miniprompts {prompt_folder} no existe")


def get_dataset(config):
    ds = config['dataset']
    return load_dataset(ds['path'], ds['sheet_name'], ds['columns'])


def get_prompts(config, visualize=True):
    prompt_folder = config['prompt_folder']
    ensure_prompt_folder(prompt_folder)
    return generate_prompts(config['prompt'], prompt_folder, visualize=visualize)


def cmd_generate(config, dataset=None):
    dataset = get_dataset(config) if dataset is None else dataset
    prompts = get_prompts(config)
    print()
    g = config['generate']
    return generate_responses(dataset, prompts, repetitions=g['repetitions'], set_size=g['set_size'],
                              seed=g['seed'], model=config['model'], temperature=config['temperature'],
                              balance_set=g.get('balance_set', False), repeat_set=g.get('repeat_set', False),
                              output_dir=config.get('responses_dir', 'Responses'))


def cmd_evaluate(config, filenames=None, prefix=None, dataset=None, normalize=False):
    dataset = get_dataset(config) if dataset is None else dataset
    ensure_prompt_folder(config['prompt_folder'])
    if filenames is None:
        filenames = get_files_starting_with(config.get('responses_dir', 'Responses'), prefix or '')
    if not filenames:
        sys.exit("Error: No se encontraron archivos de respuestas para evaluar")

    e = config['evaluate']
    return evaluate_mul(filenames, dataset, eval_function=e['eval_function'], eval_params=e.get('eval_params'),
                        train_set_size=e['train_set_size'], seed=e['seed'], model=config['model'],
                        temperature=config['temperature'], prompt_folder=config['prompt_folder'],
                        output_dir=config.get('evals_dir', 'Evals'), normalize=normalize)


def main():
    parser = argparse.ArgumentParser(description="Framework para experimentos EvaluAI")
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('prompts', help="Muestra el o los prompts definidos en la configuración")
    p.add_argument('--config', default='config.json')

    p = sub.add_parser('generate', help="Genera respuestas del modelo y las guarda en Responses/")
    p.add_argument('--config', default='config.json')

    p = sub.add_parser('evaluate', help="Ajusta/aplica la función de mapeo sobre archivos de respuestas")
    p.add_argument('--config', default='config.json')
    p.add_argument('files', nargs='*', help="Archivos de respuestas (por defecto, los de Responses/ que empiecen con --prefix)")
    p.add_argument('--prefix', default='', help="Prefijo de los archivos de respuestas a evaluar")
    p.add_argument('--normalize', action='store_true', help="Normaliza MSE y MAE entre 0 y 1")

    p = sub.add_parser('read', help="Muestra las métricas de archivos de evaluación")
    p.add_argument('files', nargs='*', help="Archivos de evaluación (por defecto, los de Evals/ que empiecen con --prefix)")
    p.add_argument('--dir', default='Evals')
    p.add_argument('--prefix', default='', help="Prefijo de los archivos de evaluación")
    p.add_argument('--normalize', action='store_true', help="Normaliza MSE y MAE entre 0 y 1")
    p.add_argument('--show', action='store_true', help="Abre el gráfico en el navegador")

    p = sub.add_parser('all', help="Ejecuta generate y luego evaluate sobre las respuestas generadas")
    p.add_argument('--config', default='config.json')
    p.add_argument('--normalize', action='store_true', help="Normaliza MSE y MAE entre 0 y 1")

    args = parser.parse_args()
    load_env()

    if args.command == 'read':
        files = args.files or get_files_starting_with(args.dir, args.prefix)
        if not files:
            sys.exit("Error: No se encontraron archivos de evaluación")
        read_evals(files, normalize=args.normalize, show_plot=args.show)
        return

    config = load_config(args.config)

    if args.command == 'prompts':
        get_prompts(config)
    elif args.command == 'generate':
        cmd_generate(config)
    elif args.command == 'evaluate':
        cmd_evaluate(config, filenames=args.files or None, prefix=args.prefix, normalize=args.normalize)
    elif args.command == 'all':
        dataset = get_dataset(config)
        filenames = cmd_generate(config, dataset)
        cmd_evaluate(config, filenames=filenames, dataset=dataset, normalize=args.normalize)


if __name__ == '__main__':
    main()
