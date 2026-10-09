# Datos

`evaluai_dataset.xlsx` contiene los datasets usados en los experimentos, con nombres de columnas estandarizados. La hoja `README` describe cada columna.

| Hoja | Contenido |
|---|---|
| `DataStructures` | 319 respuestas del curso Estructuras de Datos evaluadas por los tres expertos, antes del filtro de discrepancia. `in_paper_290` marca las 287 usadas en el paper y `exclusion_reason` el motivo de exclusión del resto. |
| `Optimization` | 535 respuestas (45 preguntas, 10 certámenes) del curso de Optimización. |
| `FewShot` | Respuestas de nivel 1/3, 2/3 y 3/3 usadas como ejemplos few-shot (no forman parte de `DataStructures`). |

El archivo se genera a partir del libro de trabajo original con:

```bash
python data/build_dataset.py datasets_v2.xlsx
```

Observaciones:
- C1-BA tenía 20 respuestas evaluadas por los tres expertos; 3 se excluyeron por discrepancia y no se conservaron. En las 17 restantes las tres notas coinciden y el libro original registra una sola (columna `grade_g2`).
- Se eliminaron las respuestas sin las tres notas humanas: tres de C3-S que no tenían ninguna (en la versión anterior del paper su puntaje de referencia provenía del evaluador automático) y tres con notas incompletas. Los `id` conservan la numeración original, y el código deriva de ellos el número de fila con que las corridas guardadas identifican cada respuesta.
- Con tres notas que difieren a lo más en 1, el promedio redondeado coincide con la moda.
