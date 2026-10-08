import random

import numpy as np
import pandas as pd

from .metrics import metrics_per_repetition
from .optimization import optimize_params, convert_gpt_scores

SUMMARY_COLS = ['macro_mse', 'micro_mse', 'mae', 'r2', 'bias', 'PTB', 'RTB', 'PBB', 'RBB']


# Aplica una función de mapeo con parámetros dados a un archivo de respuestas (puntajes crudos 0-10)
def apply_mapping(response_df, criteria, eval_function, params):
    df = response_df.copy()
    criteria_scores = df[criteria].astype(float).values.tolist()
    df['gpt_eval'] = convert_gpt_scores(criteria_scores, None, eval_function, params)
    return df


# Evalúa una grilla de umbrales (a, b) para map2-simple sin nuevas consultas al modelo
def sensitivity_grid(response_df, criteria, a_values, b_values, normalize=True):
    weights = [1.0 / len(criteria)] * len(criteria)
    rows = []
    for a in a_values:
        for b in b_values:
            if b <= a: continue
            df = apply_mapping(response_df, criteria, 'map2-simple', weights + [a, b])
            stats = metrics_per_repetition(df, normalize)[SUMMARY_COLS]
            row = {'a': a, 'b': b}
            row.update({c: stats[c].mean() for c in SUMMARY_COLS})
            row.update({c + '_std': stats[c].std(ddof=0) for c in SUMMARY_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


# Curva de calibración: para cada repetición se ajustan los umbrales con k respuestas calificadas
# tomadas del propio conjunto y se evalúa sobre las restantes, comparando con los umbrales fijos.
def calibration_curve(response_df, criteria, sizes, fixed_params, eval_function='map2-simple', seed=42, draws=3, normalize=True, objective='mse'):
    rng = random.Random(seed)
    rows = []
    for repetition, rep_df in response_df.groupby('repetition'):
        for k in sizes:
            if k >= len(rep_df): continue
            for draw in range(draws):
                calib = rep_df.sample(k, random_state=rng.randint(0, 100000))
                rest = rep_df.drop(calib.index)
                params = optimize_params(calib[criteria].astype(float).values.tolist(), calib['real_eval'].tolist(), eval_function, objective)

                fitted = metrics_per_repetition(apply_mapping(rest, criteria, eval_function, params), normalize).iloc[0]
                fixed = metrics_per_repetition(apply_mapping(rest, criteria, eval_function, fixed_params), normalize).iloc[0]
                row = {'repetition': repetition, 'k': k, 'draw': draw, 'a': params[-2], 'b': params[-1]}
                row.update({'fitted_' + c: fitted[c] for c in SUMMARY_COLS})
                row.update({'fixed_' + c: fixed[c] for c in SUMMARY_COLS})
                rows.append(row)
    return pd.DataFrame(rows)


# Histograma de puntajes crudos (0-10) del modelo para cada nivel humano
# MSE (escala 0-1) entre cada par de evaluadores humanos y entre el modelo y cada evaluador
def grader_agreement(eval_df, grades_df, graders):
    df = eval_df[['row', 'gpt_eval']].merge(grades_df[['row'] + graders], on='row')
    names = graders + ['model']
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    counts = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b: continue
            col_a = df['gpt_eval'] if a == 'model' else df[a]
            col_b = df['gpt_eval'] if b == 'model' else df[b]
            mask = col_a.notna() & col_b.notna()
            table.loc[a, b] = (((col_a[mask] - col_b[mask]) / 3) ** 2).mean()
            counts.loc[a, b] = mask.sum()
    return table, counts


def raw_score_distribution(response_df, criteria):
    score = response_df[criteria].astype(float).mean(axis=1).round().astype(int)
    table = pd.crosstab(response_df['real_eval'], score, normalize='index')
    table.index.name = 'real_eval'
    table.columns.name = 'raw_score'
    return table


# Gráfico estático de la distribución de puntajes del modelo por respuesta, agrupadas por puntaje real.
# Cada punto es una respuesta: media (y desviación estándar) de su puntaje mapeado en los conjuntos de prueba en que aparece.
def distribution_plot(eval_df, output_file, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_row = eval_df.groupby('row').agg(real_eval=('real_eval', 'first'), dataset=('dataset', 'first'),
                                         mean=('gpt_eval', 'mean'), sd=('gpt_eval', lambda x: x.std(ddof=0)))
    per_row = per_row.sort_values(['real_eval', 'mean']).reset_index()
    counts = per_row['real_eval'].value_counts()
    per_row['x'] = per_row.groupby('real_eval').cumcount()
    per_row['x'] = per_row['real_eval'] + (per_row['x'] + 1) / per_row['real_eval'].map(counts).add(1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for level in range(4):
        ax.hlines(level, level, level + 1, colors='red', linewidth=1)
    markers = ['o', 's', 'D', '^', 'v', 'P']
    for i, (name, g) in enumerate(sorted(per_row.groupby('dataset'), key=lambda t: str(t[0]))):
        ax.errorbar(g['x'], g['mean'], yerr=g['sd'], fmt=markers[i % len(markers)], markersize=4, capsize=2,
                    elinewidth=0.8, alpha=0.85, label=str(name))
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 3.1)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], ['0/3', '1/3', '2/3', '3/3'])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel('Reference score')
    ax.set_ylabel('Model score (mapped, 0-3)')
    if title: ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Dataset', loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
