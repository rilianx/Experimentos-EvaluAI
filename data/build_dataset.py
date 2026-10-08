"""Construye data/evaluai_dataset.xlsx a partir del libro de trabajo original (datasets_v2.xlsx).

    python data/build_dataset.py datasets_v2.xlsx

Hojas de origen:
  - "AllDatasets ": todas las respuestas de Estructuras de Datos antes del filtro (con las notas de cada evaluador)
  - "AllDatasets (1dif)": las 290 respuestas usadas en el paper (con conocimiento simple y detallado)
  - "Datasets Profesora Leslie": respuestas del curso de Optimización
"""
import sys

import numpy as np
import pandas as pd

SUBSETS = {  # nombre en el libro original -> nombre en el paper
    'C3-Sample100': ('C3-S', 'C3'),
    'C2-Sample100': ('C2-Smax', 'C2'),
    'C2-Nan': ('C2-hard', 'C2'),
    'C1-OscarBadAnswers20': ('C1-BA', 'C1'),
}
GRADERS = {'Eval Profe': 'grade_g1', 'Eval Oscar': 'grade_g2', 'Eval Felipe': 'grade_g3'}
GRADE_COLS = list(GRADERS.values())


def norm(text):
    return " ".join(str(text).split())


def key(df):
    return df['Pregunta'].map(norm) + '||' + df['Respuesta'].map(norm)


def question_ids(questions, prefix):
    ids = {q: f"{prefix}{i + 1:03d}" for i, q in enumerate(dict.fromkeys(questions.map(norm)))}
    return questions.map(lambda q: ids[norm(q)])


def build_datastructures(book):
    all_df = pd.read_excel(book, sheet_name='AllDatasets ')
    final = pd.read_excel(book, sheet_name='AllDatasets (1dif)')
    all_df['key'], final['key'] = key(all_df), key(final)
    final = final.set_index('key')

    df = pd.DataFrame({
        'source_sheet': all_df['DatasetProveniente'],
        'subset': all_df['DatasetProveniente'].map(lambda s: SUBSETS[s][0]),
        'assessment': all_df['DatasetProveniente'].map(lambda s: SUBSETS[s][1]),
        'question': all_df['Pregunta'].map(str.strip),
        'answer': all_df['Respuesta'].map(lambda a: str(a).strip()),
        'knowledge_detailed': all_df['Contexto'],
        'knowledge_simple': all_df['key'].map(final['Contexto simple']),
    })
    for src, dst in GRADERS.items():
        df[dst] = all_df[src]
    # Puntaje del evaluador automático anterior (solo existe para C3)
    df['earlier_gpt_score'] = all_df['Eval'].where(df['assessment'] != 'C2')
    df['key'] = all_df['key']

    # C1-BA fue evaluado por un único experto (columna Eval de su hoja de origen)
    c1 = df['subset'] == 'C1-BA'
    df.loc[c1, 'grade_g2'] = all_df.loc[c1, 'key'].map(final['Eval'])
    df.loc[c1, 'earlier_gpt_score'] = np.nan  # En el libro original es una copia de la nota del experto

    grades = df[GRADE_COLS]
    df['n_human_grades'] = grades.notna().sum(axis=1)
    df['grade_spread'] = grades.max(axis=1) - grades.min(axis=1)
    df['score_mean'] = grades.mean(axis=1)
    df['score'] = df['score_mean'].round().astype('Int64')
    df['in_paper_290'] = all_df['key'].isin(final.index)

    reasons = []
    for _, row in df.iterrows():
        if row['n_human_grades'] == 0:
            reasons.append('no human grades')
        elif row['subset'] != 'C1-BA' and row['n_human_grades'] < 3 and not row['in_paper_290']:
            reasons.append('incomplete grades')
        elif row['grade_spread'] > 1:
            reasons.append('grader disagreement > 1/3')
        else:
            reasons.append('')
    df['exclusion_reason'] = reasons

    # Filas del paper sin notas humanas: su puntaje de referencia venía del evaluador automático
    gpt_ref = df['in_paper_290'] & (df['n_human_grades'] == 0)
    df['reference_source'] = np.where(df['subset'] == 'C1-BA', 'single expert', 'human graders')
    df.loc[gpt_ref, 'reference_source'] = 'earlier GPT score'
    df.loc[gpt_ref, 'score'] = all_df.loc[gpt_ref, 'key'].map(final['Promedio Redondeado'])
    df.loc[df['in_paper_290'] & (df['n_human_grades'] == 2), 'reference_source'] = 'human graders (2 of 3)'

    order = ['C3-S', 'C2-Smax', 'C2-hard', 'C1-BA']
    df['subset'] = pd.Categorical(df['subset'], order, ordered=True)
    df = df.sort_values(['subset', 'question', 'answer'], kind='stable').reset_index(drop=True)
    df['subset'] = df['subset'].astype(str)
    df.insert(0, 'id', [f"DS-{i + 1:03d}" for i in range(len(df))])
    df.insert(4, 'question_id', question_ids(df['question'], 'DSQ'))

    paper = df[df['in_paper_290']]
    mismatch = (paper['score'] != paper['key'].map(final['Promedio Redondeado'])).sum()
    assert mismatch == 0, f"{mismatch} puntajes no coinciden con el libro original"
    return df.drop(columns=['key'])


def build_optimization(book):
    src = pd.read_excel(book, sheet_name='Datasets Profesora Leslie')
    df = pd.DataFrame({
        'exam': src['DataSet'],
        'question': src['Pregunta'].map(str.strip),
        'answer': src['Respuesta'].map(lambda a: str(a).strip()),
        'knowledge_detailed': src['Contexto detallado'],
        'points': src['Eval'],
        'max_points': src['Max Puntaje'],
        'score': src['Eval Round'].astype(int),
    })
    assert ((df['points'] / df['max_points'] * 3).round() == df['score']).all()
    df = df.sort_values(['exam', 'question', 'answer'], kind='stable').reset_index(drop=True)
    df.insert(0, 'id', [f"OPT-{i + 1:03d}" for i in range(len(df))])
    df.insert(2, 'question_id', question_ids(df['question'], 'OPQ'))
    return df


FEWSHOT = [  # (nivel, hoja de origen, fila en esa hoja) de las respuestas usadas como ejemplos
    (1, 'C1-balanced20', 7), (1, 'C2-claim', 69),
    (2, 'C2-claim', 109), (2, 'C2-claim', 33),
    (3, 'C2-claim', 34), (3, 'C3-G57', 17),
]


def build_fewshot(book):
    rows = []
    for level, sheet, row in FEWSHOT:
        r = pd.read_excel(book, sheet_name=sheet).iloc[row - 2]
        rows.append({'level': level, 'source_sheet': sheet, 'source_row': row,
                     'question': str(r['Pregunta']).strip(), 'answer': str(r['Respuesta']).strip()})
    return pd.DataFrame(rows)


README = [
    ('Sheet', 'Column', 'Description'),
    ('DataStructures', '', 'Respuestas del curso Estructuras de Datos (ICI/ICD, PUCV) antes y después del filtro. Usar in_paper_290 / exclusion_reason para reproducir el dataset del paper.'),
    ('DataStructures', 'id', 'Identificador de la respuesta'),
    ('DataStructures', 'subset', 'Subconjunto según el paper: C3-S, C2-Smax, C2-hard, C1-BA'),
    ('DataStructures', 'source_sheet', 'Nombre del subconjunto en el libro de trabajo original'),
    ('DataStructures', 'assessment', 'Control del que proviene la respuesta (C1, C2, C3)'),
    ('DataStructures', 'question_id', 'Identificador de la pregunta'),
    ('DataStructures', 'knowledge_detailed / knowledge_simple', 'Párrafo de conocimiento (no visible para el estudiante), versión detallada y simple'),
    ('DataStructures', 'grade_g1, grade_g2, grade_g3', 'Notas (0-3) de los evaluadores humanos 1, 2 y 3. C1-BA tiene un único experto (grade_g2)'),
    ('DataStructures', 'earlier_gpt_score', 'Puntaje (0-3) de una versión anterior del evaluador automático; solo disponible para C3'),
    ('DataStructures', 'n_human_grades, grade_spread', 'Cantidad de notas humanas y diferencia máxima entre ellas'),
    ('DataStructures', 'score_mean, score', 'Promedio de las notas humanas y puntaje de referencia (promedio redondeado; coincide con la moda cuando la diferencia es <= 1)'),
    ('DataStructures', 'in_paper_290', 'La respuesta forma parte de las 290 usadas en el paper'),
    ('DataStructures', 'exclusion_reason', 'Motivo de exclusión de las respuestas que no forman parte del paper'),
    ('DataStructures', 'reference_source', 'Origen del puntaje de referencia. "earlier GPT score": la respuesta no tiene notas humanas y su referencia proviene del evaluador automático (se recomienda excluirla)'),
    ('Optimization', '', 'Respuestas de certámenes del curso de Optimización (2020-2024)'),
    ('Optimization', 'exam', 'Certamen de origen'),
    ('Optimization', 'knowledge_detailed', 'Respuesta correcta / pauta'),
    ('Optimization', 'points, max_points', 'Puntaje obtenido y puntaje máximo de la pregunta'),
    ('Optimization', 'score', 'Puntaje escalado a 0-3 y redondeado (round(points / max_points * 3))'),
    ('FewShot', '', 'Respuestas de niveles intermedios y completos usadas como ejemplos few-shot (no forman parte de DataStructures). Evaluadas por un único experto.'),
]


def main():
    book = pd.ExcelFile(sys.argv[1])
    output = sys.argv[2] if len(sys.argv) > 2 else 'data/evaluai_dataset.xlsx'
    ds, opt, few = build_datastructures(book), build_optimization(book), build_fewshot(book)

    with pd.ExcelWriter(output) as writer:
        pd.DataFrame(README[1:], columns=README[0]).to_excel(writer, sheet_name='README', index=False)
        ds.to_excel(writer, sheet_name='DataStructures', index=False)
        opt.to_excel(writer, sheet_name='Optimization', index=False)
        few.to_excel(writer, sheet_name='FewShot', index=False)

    paper = ds[ds['in_paper_290']]
    print(f"DataStructures: {len(ds)} respuestas, {len(paper)} en el paper")
    print(paper['subset'].value_counts().to_dict(), paper['score'].value_counts().sort_index().to_dict())
    print("Excluidas:", ds.loc[~ds['in_paper_290'], 'exclusion_reason'].value_counts().to_dict())
    print("Origen de la referencia (paper):", paper['reference_source'].value_counts().to_dict())
    print(f"Optimization: {len(opt)} respuestas, {opt['question_id'].nunique()} preguntas, {opt['exam'].nunique()} certámenes")
    print(f"Guardado en {output}")


if __name__ == '__main__':
    main()
