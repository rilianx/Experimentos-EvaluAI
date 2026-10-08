"""Ejecuta una suite de experimentos definida en un JSON y guarda las métricas agregadas.

    python run_suite.py experiments/datastructures.json
    python run_suite.py experiments/datastructures.json --only base,knowledge_detailed
    python run_suite.py experiments/datastructures.json --smoke     # corrida mínima para probar la configuración

Las salidas completas (respuestas del modelo, con las respuestas de los estudiantes) quedan en runs/<suite>/.
Las métricas agregadas, sin datos de estudiantes, quedan en results/<suite>/.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

from evaluai import load_dataset, generate_prompts, generate_responses
from evaluai.experiments import evaluate
from evaluai.metrics import metrics_per_repetition, paired_comparison, stability_metrics
from evaluai.analysis import sensitivity_grid, calibration_curve, raw_score_distribution, SUMMARY_COLS
from main import load_env, ensure_prompt_folder


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Suite:
    def __init__(self, path, smoke=False):
        with open(path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.name = self.config['name'] + ('-smoke' if smoke else '')
        self.smoke = smoke
        self.run_dir = os.path.join('runs', self.name)
        self.results_dir = os.path.join('results', self.name)
        self.experiments = {e['name']: e for e in self.config['experiments']}
        self._datasets = {}
        self.evals = {}
        self.responses = {}

    def exp_config(self, name):
        exp = self.experiments[name]
        if 'responses_from' in exp:
            base = self.exp_config(exp['responses_from'])
            cfg = deep_merge(base, {k: v for k, v in exp.items() if k not in ('prompt', 'responses_from')})
        else:
            cfg = deep_merge({k: v for k, v in self.config.items() if k not in ('experiments', 'comparisons', 'analyses')}, exp)
        if self.smoke:
            cfg['generate'].update(repetitions=2, set_size=12)
            cfg['evaluate']['train_set_size'] = 12
        return cfg

    def dataset(self, cfg):
        ds = cfg['dataset']
        key = json.dumps(ds, sort_keys=True)
        if key not in self._datasets:
            self._datasets[key] = load_dataset(ds['path'], ds['sheet_name'], ds['columns'], verbose=False)
        return self._datasets[key]

    def exp_dir(self, name):
        return os.path.join(self.run_dir, name)

    def run(self, name):
        exp = self.experiments[name]
        cfg = self.exp_config(name)
        print(f"\n===== {name} =====")
        dataset = self.dataset(cfg)
        ensure_prompt_folder(cfg['prompt_folder'])

        if 'responses_from' in exp:
            responses_file = self.responses[exp['responses_from']]
        else:
            prompts = generate_prompts(cfg['prompt'], cfg['prompt_folder'], visualize=False)
            if len(prompts) != 1:
                sys.exit(f"Error: El experimento {name} no debe usar el comodín *")
            g = cfg['generate']
            responses_file = generate_responses(dataset, prompts, repetitions=g['repetitions'], set_size=g['set_size'],
                                                seed=g['seed'], model=cfg['model'], temperature=cfg['temperature'],
                                                balance_set=g.get('balance_set', False), repeat_set=g.get('repeat_set', False),
                                                output_dir=os.path.join(self.exp_dir(name), 'Responses'))[0]
        self.responses[name] = responses_file

        e = cfg['evaluate']
        eval_file = evaluate(responses_file, dataset, e['eval_function'], e.get('eval_params'), e['train_set_size'], e['seed'],
                             cfg['model'], cfg['temperature'], cfg['prompt_folder'], os.path.join(self.exp_dir(name), 'Evals'))
        self.evals[name] = pd.read_excel(eval_file, sheet_name='Evaluation')

    def criteria(self, name):
        cfg = self.exp_config(name)
        return generate_prompts(cfg['prompt'], cfg['prompt_folder'], visualize=False)[0].criteria

    def response_df(self, name):
        return pd.read_excel(self.responses[name], sheet_name='Responses')

    def summarize(self):
        os.makedirs(self.results_dir, exist_ok=True)
        means, stds = [], []
        for name, eval_df in self.evals.items():
            stats = metrics_per_repetition(eval_df, normalize=True).drop(columns=['repetition'])
            mean = stats.mean().to_dict()
            std = stats.std(ddof=0).to_dict()
            params = eval_df['params'].iloc[0] if eval_df['params'].nunique() == 1 else 'per-repetition'
            means.append({'experiment': name, 'params': params, **mean})
            stds.append({'experiment': name, **std})
        means, stds = pd.DataFrame(means), pd.DataFrame(stds)
        means.to_csv(os.path.join(self.results_dir, 'summary_mean.csv'), index=False)
        stds.to_csv(os.path.join(self.results_dir, 'summary_std.csv'), index=False)

        cols = ['macro_mse', 'mse_0', 'mse_1', 'mse_2', 'mse_3', 'micro_mse', 'mae', 'r2', 'bias', 'PTB', 'RTB', 'PBB', 'RBB']
        lines = ['| experiment | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
        for (_, m), (_, s) in zip(means.iterrows(), stds.iterrows()):
            lines.append(f"| {m['experiment']} | " + ' | '.join(f"{m[c]:.3f} ± {s[c]:.3f}" for c in cols) + ' |')
        with open(os.path.join(self.results_dir, 'summary.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))

    def compare(self):
        rows = []
        for a, b in self.config.get('comparisons', []):
            if a in self.evals and b in self.evals:
                rows.append({'a': a, 'b': b, **paired_comparison(self.evals[a], self.evals[b])})
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(self.results_dir, 'comparisons.csv'), index=False)
            print('\nComparaciones pareadas (Wilcoxon):')
            print(df.to_string(index=False))

    def analyze(self):
        analyses = self.config.get('analyses', {})

        name = analyses.get('stability')
        if name in self.responses:
            stats = stability_metrics(self.response_df(name), self.criteria(name))
            eval_stats = metrics_per_repetition(self.evals[name])
            per_rep = self.evals[name].groupby('row')['gpt_eval'].std(ddof=0)
            stats['mean_sd_mapped_0_1'] = (per_rep / 3).mean()
            stats['macro_mse_sd_across_runs'] = eval_stats['macro_mse'].std(ddof=0)
            pd.DataFrame([stats]).to_csv(os.path.join(self.results_dir, 'stability.csv'), index=False)
            print('\nEstabilidad:', stats)

        for name in analyses.get('raw_distribution', []):
            if name in self.responses:
                table = raw_score_distribution(self.response_df(name), self.criteria(name))
                table.to_csv(os.path.join(self.results_dir, f'raw_distribution_{name}.csv'))

        sens = analyses.get('sensitivity')
        if sens and sens['experiment'] in self.responses:
            name = sens['experiment']
            a_values = np.arange(*sens.get('a_range', [0, 5.01, 0.5]))
            b_values = np.arange(*sens.get('b_range', [5, 10.01, 0.5]))
            grid = sensitivity_grid(self.response_df(name), self.criteria(name), a_values, b_values)
            grid.to_csv(os.path.join(self.results_dir, 'sensitivity.csv'), index=False)
            plot_sensitivity(grid, os.path.join(self.results_dir, 'sensitivity.png'))

        cal = analyses.get('calibration')
        if cal and cal['experiment'] in self.responses:
            name = cal['experiment']
            sizes = [k for k in cal.get('sizes', [5, 10, 20, 40, 60]) if not self.smoke or k < 8]
            curve = calibration_curve(self.response_df(name), self.criteria(name), sizes, cal['fixed_params'],
                                      draws=1 if self.smoke else cal.get('draws', 3))
            curve.to_csv(os.path.join(self.results_dir, 'calibration.csv'), index=False)
            agg = curve.groupby('k')[[c for c in curve.columns if c.startswith(('fitted_', 'fixed_'))]].agg(['mean', 'std'])
            agg.to_csv(os.path.join(self.results_dir, 'calibration_summary.csv'))


def plot_sensitivity(grid, output_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ['macro_mse', 'micro_mse']):
        pivot = grid.pivot(index='b', columns='a', values=metric).sort_index(ascending=False)
        im = ax.imshow(pivot.values, cmap='viridis_r', aspect='auto',
                       extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
        ax.set_xlabel('a (lower threshold)')
        ax.set_ylabel('b (upper threshold)')
        ax.set_title(metric.replace('_', '-').upper())
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta una suite de experimentos EvaluAI")
    parser.add_argument('suite')
    parser.add_argument('--only', default='', help="Lista de experimentos separados por coma (por defecto, todos)")
    parser.add_argument('--smoke', action='store_true', help="Corrida mínima (2 repeticiones, 12 respuestas)")
    args = parser.parse_args()
    load_env()

    suite = Suite(args.suite, smoke=args.smoke)
    names = [n for n in args.only.split(',') if n] or list(suite.experiments)
    for name in names:
        if name not in suite.experiments:
            sys.exit(f"Error: El experimento {name} no existe en la suite")
        dep = suite.experiments[name].get('responses_from')
        if dep and dep not in names:
            sys.exit(f"Error: El experimento {name} requiere ejecutar también {dep}")

    for name in suite.experiments:
        if name in names:
            suite.run(name)

    suite.summarize()
    suite.compare()
    suite.analyze()
    print(f"\nResultados agregados en {suite.results_dir}/")


if __name__ == '__main__':
    main()
