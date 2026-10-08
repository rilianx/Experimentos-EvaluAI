# Datos

`evaluai_dataset.xlsx` contiene los datasets usados en los experimentos, con nombres de columnas estandarizados. La hoja `README` describe cada columna.

| Hoja | Contenido |
|---|---|
| `DataStructures` | 325 respuestas del curso Estructuras de Datos antes del filtro, con las notas de los tres evaluadores. `in_paper_290` marca las 290 usadas en el paper y `exclusion_reason` el motivo de exclusión del resto. |
| `Optimization` | 535 respuestas (45 preguntas, 10 certámenes) del curso de Optimización. |
| `FewShot` | Respuestas de nivel 1/3, 2/3 y 3/3 usadas como ejemplos few-shot (no forman parte de `DataStructures`). |

El archivo se genera a partir del libro de trabajo original con:

```bash
python data/build_dataset.py datasets_v2.xlsx
```

Observaciones:
- C1-BA fue evaluado por un único experto (columna `grade_g2`), por lo que el filtro de discrepancia entre evaluadores no aplica.
- Tres respuestas de C3-S (`reference_source = "earlier GPT score"`) no tienen notas humanas: su puntaje de referencia en el paper provenía del evaluador automático anterior. Las suites de `experiments/` las excluyen.
- Con tres notas que difieren a lo más en 1, el promedio redondeado coincide con la moda.
