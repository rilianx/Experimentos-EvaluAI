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
def calibration_curve(response_df, criteria, sizes, fixed_params, eval_function='map2-simple', seed=42, draws=3, normalize=True):
    rng = random.Random(seed)
    rows = []
    for repetition, rep_df in response_df.groupby('repetition'):
        for k in sizes:
            if k >= len(rep_df): continue
            for draw in range(draws):
                calib = rep_df.sample(k, random_state=rng.randint(0, 100000))
                rest = rep_df.drop(calib.index)
                params = optimize_params(calib[criteria].astype(float).values.tolist(), calib['real_eval'].tolist(), eval_function)

                fitted = metrics_per_repetition(apply_mapping(rest, criteria, eval_function, params), normalize).iloc[0]
                fixed = metrics_per_repetition(apply_mapping(rest, criteria, eval_function, fixed_params), normalize).iloc[0]
                row = {'repetition': repetition, 'k': k, 'draw': draw, 'a': params[-2], 'b': params[-1]}
                row.update({'fitted_' + c: fitted[c] for c in SUMMARY_COLS})
                row.update({'fixed_' + c: fixed[c] for c in SUMMARY_COLS})
                rows.append(row)
    return pd.DataFrame(rows)


# Histograma de puntajes crudos (0-10) del modelo para cada nivel humano
def raw_score_distribution(response_df, criteria):
    score = response_df[criteria].astype(float).mean(axis=1).round().astype(int)
    table = pd.crosstab(response_df['real_eval'], score, normalize='index')
    table.index.name = 'real_eval'
    table.columns.name = 'raw_score'
    return table
