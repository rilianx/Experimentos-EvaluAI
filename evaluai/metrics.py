import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score)

LEVELS = [0, 1, 2, 3]


# Métricas por bandas: la banda superior (inferior) contiene la fracción de predicciones más altas (bajas)
# igual a la proporción de respuestas con puntaje humano 3/3 (0/3)
def band_metrics(real, pred):
    real, pred = np.asarray(real), np.asarray(pred)
    stats = {}
    for name, level, top in (('TB', 3, True), ('BB', 0, False)):
        c = np.mean(real == level)
        if c == 0:
            stats['P' + name], stats['R' + name] = np.nan, np.nan
            continue
        if top:
            band = pred >= np.quantile(pred, 1 - c)
        else:
            band = pred <= np.quantile(pred, c)
        hits = np.sum(band & (real == level))
        stats['P' + name] = hits / band.sum()
        stats['R' + name] = hits / np.sum(real == level)
    return stats


# Calcula las métricas de una repetición. Con normalize, los errores se expresan en escala 0-1
def compute_metrics(df_rep, normalize=True):
    k = 3 if normalize else 1
    real, pred = df_rep['real_eval'] / k, df_rep['gpt_eval'] / k

    mse_per_class = {}
    bias_per_class = {}
    for level in LEVELS:
        mask = df_rep['real_eval'] == level
        if mask.any():
            mse_per_class[level] = mean_squared_error(real[mask], pred[mask])
            bias_per_class[level] = float(np.mean(pred[mask] - real[mask]))

    stats = {
        'macro_mse': np.mean(list(mse_per_class.values())),
        'micro_mse': mean_squared_error(real, pred),
        'mae': mean_absolute_error(real, pred),
        'r2': r2_score(real, pred),
        'spearman': spearmanr(df_rep['real_eval'], df_rep['gpt_eval'])[0],
        'bias': float(np.mean(pred - real)),
    }
    for level in LEVELS:
        stats[f'mse_{level}'] = mse_per_class.get(level, np.nan)
    for level in LEVELS:
        stats[f'bias_{level}'] = bias_per_class.get(level, np.nan)
    for name, g in df_rep.groupby('dataset'):
        stats[f'mse_{name}'] = mean_squared_error(g['real_eval'] / k, g['gpt_eval'] / k)

    stats.update(band_metrics(df_rep['real_eval'], df_rep['gpt_eval']))

    # Métricas de clasificación: Se calculan con puntaje GPT redondeado (0-3)
    rounded = df_rep['gpt_eval'].round().astype(int)
    stats.update({
        'accuracy': accuracy_score(df_rep['real_eval'], rounded),
        'precision': precision_score(df_rep['real_eval'], rounded, average='weighted', zero_division=0),
        'recall': recall_score(df_rep['real_eval'], rounded, average='weighted', zero_division=0),
        'f1': f1_score(df_rep['real_eval'], rounded, average='weighted', zero_division=0),
    })
    return stats


# Métricas por repetición de un conjunto de evaluaciones
def metrics_per_repetition(eval_df, normalize=True):
    rows = []
    for repetition, df_rep in eval_df.groupby('repetition'):
        stats = compute_metrics(df_rep, normalize)
        stats['repetition'] = repetition
        rows.append(stats)
    return pd.DataFrame(rows)


# Compara dos configuraciones evaluadas sobre los mismos conjuntos de prueba (misma semilla).
# Usa Wilcoxon pareado sobre el error cuadrático de cada respuesta y sobre el Macro-MSE de cada repetición.
def paired_comparison(eval_a, eval_b, normalize=True):
    k = 3 if normalize else 1
    keys = ['repetition', 'row']
    merged = eval_a[keys + ['real_eval', 'gpt_eval']].merge(eval_b[keys + ['gpt_eval']], on=keys, suffixes=('_a', '_b'))
    se_a = ((merged['gpt_eval_a'] - merged['real_eval']) / k) ** 2
    se_b = ((merged['gpt_eval_b'] - merged['real_eval']) / k) ** 2

    ma = metrics_per_repetition(eval_a, normalize).set_index('repetition')['macro_mse']
    mb = metrics_per_repetition(eval_b, normalize).set_index('repetition')['macro_mse']
    common = ma.index.intersection(mb.index)

    def safe_wilcoxon(x, y):
        try:
            return wilcoxon(x, y).pvalue
        except ValueError:
            return np.nan

    return {
        'n_responses': len(merged),
        'mse_a': se_a.mean(),
        'mse_b': se_b.mean(),
        'p_response_se': safe_wilcoxon(se_a, se_b),
        'macro_mse_a': ma[common].mean(),
        'macro_mse_b': mb[common].mean(),
        'p_rep_macro_mse': safe_wilcoxon(ma[common], mb[common]) if len(common) > 1 else np.nan,
    }


# Estabilidad del puntaje de cada respuesta frente a inferencias repetidas (mismo prompt, misma respuesta)
def stability_metrics(response_df, criteria):
    score = response_df[criteria].astype(float).mean(axis=1)
    df = response_df.assign(raw_score=score)
    per_response = df.groupby('row')['raw_score'].agg(['mean', 'std', 'min', 'max', 'count'])
    per_response = per_response[per_response['count'] > 1]
    return {
        'n_responses': len(per_response),
        'runs_per_response': per_response['count'].median(),
        'mean_sd_raw': per_response['std'].mean(),
        'max_sd_raw': per_response['std'].max(),
        'pct_identical': float(np.mean(per_response['max'] == per_response['min'])),
        'pct_range_le_1': float(np.mean(per_response['max'] - per_response['min'] <= 1)),
    }
