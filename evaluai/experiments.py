import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .dataset import generate_set
from .llm import chat_gpt_multiple, extract_dicts
from .metrics import metrics_per_repetition
from .optimization import optimize_params, convert_gpt_scores
from .prompts import generate_prompts

DEFAULT_PROMPT_FOLDER = "GPTEvaluator/Experiments/Miniprompts_v2"
TIMEZONE = ZoneInfo('America/Santiago')


def timestamp():
    return datetime.now(TIMEZONE).strftime('%Y%m%d-%H%M%S')


# Retorna los archivos en 'directory' que comienzan con 'prefix'
def get_files_starting_with(directory, prefix):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith('.xlsx')
    )


# Genera las respuestas con ChatGPT
def eval_gpt(df, prompt, criteria, model, temperature):
    df_copy = df.copy()

    texts = []
    for i, row in df_copy.iterrows():
        text = prompt.format(Question=row['question'], Answer=row['answer'], Context=row['context'])
        texts.append(text)

    answers_gpt = chat_gpt_multiple(texts, model, temperature)
    gpt_dicts = extract_dicts(answers_gpt)
    clean_set(df_copy, gpt_dicts, criteria)
    df_dicts = pd.DataFrame(gpt_dicts)
    response_set = pd.concat([df_copy, df_dicts.set_index(df_copy.index)], axis=1)
    return response_set


# Elimina filas del dataset donde hubo errores en la salida GPT
def clean_set(dataset, gpt_dicts, criteria):
    for i in reversed(range(len(gpt_dicts))):
        if gpt_dicts[i] is None:
            gpt_dicts.pop(i)
            dataset.drop(dataset.index[i], inplace=True)
        elif not all(key in gpt_dicts[i] for key in criteria):
            print(gpt_dicts[i])
            gpt_dicts.pop(i)
            dataset.drop(dataset.index[i], inplace=True)


# Obtiene los puntajes reales de un dataset
def get_real_scores(dataset):
    return dataset['real_eval'].tolist()


# Prepara el set de entrenamiento y obtiene los parámetros óptimos para disminuir el error
def train(train_set, prompt, criteria, eval_function, model, temperature):
    res_df = eval_gpt(train_set, prompt, criteria, model, temperature)

    real_scores = get_real_scores(res_df)
    criteria_scores = res_df[criteria].astype(float).values.tolist()
    params = optimize_params(criteria_scores, real_scores, eval_function)
    return np.round(params, 2)


# Guarda un gráfico con los puntajes obtenidos para cada pregunta ordenadas por puntaje real (0-3)
def show_distribution(full_df, output_file, show=False):
    rep = full_df['repetition'].nunique()
    full_df = full_df.drop(columns=['params'], errors='ignore')

    info_cols = ['row', 'dataset', 'question', 'answer', 'context']
    value_cols = ['gpt_eval'] + (['score'] if 'score' in full_df.columns else [])
    extra_cols = full_df.columns.difference(['question', 'answer', 'context', 'real_eval', 'row', 'dataset', 'repetition'] + value_cols).tolist()
    for col in info_cols + extra_cols:
        if not pd.api.types.is_numeric_dtype(full_df[col]):
            full_df[col] = full_df[col].fillna('').astype(str).str.wrap(80).str.replace('\n', '<br>')

    full_df = full_df.pivot(index=['question', 'answer', 'context', 'real_eval', 'row', 'dataset'], columns='repetition', values=value_cols + extra_cols)
    full_df = full_df.sort_values('real_eval').reset_index()
    full_df.loc[full_df['dataset'] == 'C1-OscarBadAnswers20', 'dataset'] = 'C1-BadAnswers'

    value_counts = full_df['real_eval'].value_counts()
    n = np.array([value_counts.get(i, 0) for i in range(4)])
    x_pos = [x + (y+1)/(n[x]+1) for x in range(4) for y in range(n[x])]

    def plot(col, title):
        dev = full_df[col].apply(lambda row: row.std(ddof=0), axis=1).values.tolist()
        mean = full_df[col].apply(lambda row: round(row.mean(), 2), axis=1).values.tolist()

        fig = go.Figure()

        if col == 'gpt_eval':
            for i in range(4):
                fig.add_trace(go.Scatter(
                    x=[i, i + 1],
                    y=[i, i],
                    mode='lines',
                    line=dict(color='red', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))

        # Información extra
        if rep == 1:
            template_cols = info_cols + [col] + extra_cols
            customdata = list(zip(*[full_df[c] for c in info_cols], mean, *[full_df[c].iloc[:, 0] for c in extra_cols]))
        else:
            template_cols = info_cols + [col]
            customdata = list(zip(*[full_df[c] for c in info_cols], mean))

        template = ''
        for i, x in enumerate(template_cols):
            template += f'<b>{x}:</b> %{{customdata[{i}]}}<br>'
        template += '<extra></extra>'

        # Añadir los puntos, con un color por clase (dataset)
        palette = px.colors.qualitative.Plotly
        for k, cls in enumerate(sorted(full_df['dataset'].unique())):
            indices = full_df.index[full_df['dataset'] == cls].tolist()
            x_filtered = [x_pos[i] for i in indices]
            y_filtered = [mean[i] for i in indices]
            dev_filtered = [dev[i] for i in indices]
            customdata_filtered = [customdata[i] for i in indices]

            # Color barra de error
            color_hex = palette[k % len(palette)]
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            color_rgba = f"rgba({r}, {g}, {b}, 0.5)"

            fig.add_trace(go.Scatter(
                x=x_filtered,
                y=y_filtered,
                mode='markers',
                marker=dict(size=8, color=color_hex),
                error_y=dict(type='data', array=dev_filtered, visible=True, thickness=2, width=4, color=color_rgba),
                customdata=customdata_filtered,
                hovertemplate=template,
                name=cls
            ))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Real Eval',
            yaxis_title='GPT Eval',
            xaxis=dict(tickvals=[0, 1, 2, 3]),
            legend_title_text='Dataset',
            template='plotly_white',
            showlegend=True,
            width=1200,
            height=700,
            hoverlabel=dict(
                bgcolor="green",
                font_size=12
            )
        )

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"Gráfico guardado en {output_file}")
        if show: fig.show()

    plot('gpt_eval', 'Distribution of scores obtained by the model')


# Exporta las salidas GPT a un archivo EXCEL
def export_responses(response_set, prompt, label, repetitions, set_size, seed, model, temperature, balance_set, repeat_set, output_dir="Responses"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/{label}_{timestamp()}.xlsx'

    metadata = {
        'prompt': prompt,
        'repetitions': repetitions,
        'set_size': set_size,
        'seed': seed,
        'model': model,
        'temperature': temperature,
        'repeat_set': repeat_set,
        'balance_set': balance_set
    }
    md_set = pd.DataFrame.from_dict(metadata, orient='index')

    with pd.ExcelWriter(filename) as writer:
        response_set.to_excel(writer, sheet_name='Responses', index=False)
        md_set.to_excel(writer, sheet_name='Metadata', index=True, header=False)
    return filename


# Genera salidas GPT para un conjunto de datos. Retorna la lista de archivos generados
def generate_responses(dataset, prompts, repetitions, set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1, balance_set=False, repeat_set=False, output_dir="Responses"):
    sets = []
    for i in range(repetitions):
        set_seed = seed if repeat_set else seed + i
        sets.append(generate_set(dataset, set_size, set_seed, balance=balance_set, exclude_set=None))

    filenames = []
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data.prompt
        criteria = prompt_data.criteria
        response_set = pd.DataFrame()

        for j in range(repetitions):
            print(f"Generando respuestas para Prompt {i+1} / Conjunto {j+1}")
            rep_set = eval_gpt(sets[j], prompt, criteria, model, temperature)
            rep_set['repetition'] = j+1

            response_set = pd.concat([response_set, rep_set], ignore_index=True)
            print()

        prompt_structure = json.dumps(prompt_data.base_structure())
        label = getattr(prompt_data, 'label', 'prompt')
        filename = export_responses(response_set, prompt_structure, label, repetitions, set_size, seed, model, temperature, balance_set, repeat_set, output_dir)
        print(f"Respuestas guardadas en {filename}\n")
        filenames.append(filename)

    return filenames


# Lee y muestra resultados de archivos de evaluación. Retorna las tablas de promedios y desviaciones
def read_evals(filenames, normalize=False, plot_dir="Plots", show_plot=False):
    df_mean_all = pd.DataFrame()
    df_std_all = pd.DataFrame()

    for filename in filenames:
        full_df = pd.read_excel(filename, sheet_name='Evaluation')
        md = pd.read_excel(filename, sheet_name='Metadata (res)', header=None, index_col=0).T.reset_index(drop=True)
        mp_stats = metrics_per_repetition(full_df, normalize).drop(columns=['repetition'])
        df_mean = mp_stats.mean().to_frame().T
        df_std = mp_stats.std(ddof=0).to_frame().T

        name = os.path.basename(filename)
        for table in (df_mean, df_std):
            table.insert(0, 'prompt', md['prompt'].iloc[0])
            table.insert(0, 'file', name)
        df_mean_all = pd.concat([df_mean_all, df_mean], ignore_index=True)
        df_std_all = pd.concat([df_std_all, df_std], ignore_index=True)

    with pd.option_context('display.max_columns', None, 'display.max_colwidth', None, 'display.width', None):
        print("\nTabla Promedios")
        print(df_mean_all.to_string())
        print("\nTabla Desviación estándar")
        print(df_std_all.to_string())

    repeat_test_set = str(md['repeat_set'].iloc[0]).lower() == 'true'
    repetitions = int(md['repetitions'].iloc[0])
    if len(filenames) == 1 and (repeat_test_set or repetitions == 1):
        print()
        plot_file = os.path.join(plot_dir, os.path.splitext(os.path.basename(filenames[0]))[0] + '.html')
        show_distribution(full_df, plot_file, show_plot)

    return df_mean_all, df_std_all


# Exportar archivo de evaluaciones
def export_eval(filename, eval_set, res_md, eval_function, eval_params, train_set_size, seed, model, temperature, output_dir="Evals"):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(filename))[0]

    if eval_params is None:
        filename = f'{output_dir}/T_{filename}_eval.xlsx'
    else:
        filename = f'{output_dir}/{filename}_eval.xlsx'

    eval_md = {
        'eval_function': eval_function,
        'eval_params': eval_params,
        'train_set_size': train_set_size,
        'seed': seed,
        'model': model,
        'temperature': temperature
    }
    eval_md = pd.DataFrame.from_dict(eval_md, orient='index')

    with pd.ExcelWriter(filename) as writer:
        eval_set.to_excel(writer, sheet_name='Evaluation', index=False)
        eval_md.to_excel(writer, sheet_name='Metadata (eval)', index=True, header=False)
        res_md.to_excel(writer, sheet_name='Metadata (res)', index=True, header=False)
    return filename


# Evalúa en base a las salidas de GPT
def evaluate(filename, dataset, eval_function, eval_params=None, train_set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1, prompt_folder=DEFAULT_PROMPT_FOLDER, output_dir="Evals"):
    res_df = pd.read_excel(filename, sheet_name='Responses')

    res_md = pd.read_excel(filename, sheet_name='Metadata', header=None, index_col=0)
    prompt_dict = json.loads(res_md.T.reset_index(drop=True)['prompt'][0])
    prompt = generate_prompts(prompt_dict, prompt_folder, visualize=False)[0]

    # Dataset final de evaluaciones
    eval_set = pd.DataFrame()

    for repetition in res_df['repetition'].unique():
        rep_df = res_df[res_df['repetition'] == repetition]
        test_set = rep_df[rep_df['row'].isin(dataset['row'])].copy()

        # ENTRENAMIENTO
        rep_params = eval_params
        if eval_params is None:
            print(f"Evaluando conjunto de prueba {repetition} con AJUSTE")
            train_set = generate_set(dataset, train_set_size, int(seed + repetition), balance=False, exclude_set=test_set)
            rep_params = train(train_set, prompt.prompt, prompt.criteria, eval_function, model, temperature)
            str_params = "[" + ", ".join("{:.2f}".format(param) for param in rep_params) + "]"
            print(f"Parámetros obtenidos: {str_params}")

        criteria_scores = test_set[prompt.criteria].astype(float).values.tolist()

        test_set['gpt_eval'] = convert_gpt_scores(criteria_scores, None, eval_function, rep_params)
        test_set['params'] = ", ".join("{:.2f}".format(param) for param in rep_params)

        eval_set = pd.concat([eval_set, test_set], ignore_index=True)

    if eval_params is None: print()
    stored_params = None if eval_params is None else list(eval_params)
    return export_eval(filename, eval_set, res_md, eval_function, stored_params, train_set_size, seed, model, temperature, output_dir)


# Evalúa múltiples archivos de respuestas y muestra sus resultados. Retorna los archivos de evaluación
def evaluate_mul(filenames, dataset, eval_function, eval_params=None, train_set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1, prompt_folder=DEFAULT_PROMPT_FOLDER, output_dir="Evals", normalize=False):
    eval_files = []
    for filename in filenames:
        print(f"EVALUANDO {filename}")
        eval_file = evaluate(filename, dataset, eval_function, eval_params, train_set_size, seed, model, temperature, prompt_folder, output_dir)
        print(f"Evaluación guardada en {eval_file}\n")
        eval_files.append(eval_file)

    if len(eval_files) > 0:
        read_evals(eval_files, normalize=normalize)
    return eval_files
