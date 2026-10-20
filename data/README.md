# Data

`evaluai_dataset.xlsx` contains the datasets used in the experiments, with standardized column names. The `README` sheet describes each column.

| Sheet | Content |
|---|---|
| `DataStructures` | 319 responses from the Data Structures course graded by the three experts, before the disagreement filter. `in_paper_290` marks the 287 responses used in the paper, and `exclusion_reason` gives the reason for excluding the others. |
| `Optimization` | 535 responses (45 questions, 10 exams) from the Optimization course. |
| `FewShot` | Responses graded 1/3, 2/3 and 3/3 used as few-shot examples (not part of `DataStructures`). |

The file is built from the original workbook with:

```bash
python data/build_dataset.py datasets_v2.xlsx
```

Notes:
- C1-BA had 20 responses graded by the three experts; 3 were excluded for disagreement and were not kept. The three grades of the remaining 17 coincide; the original workbook records a single grade, which is assigned to the three graders.
- Responses without the three human grades were removed: three C3-S responses with no human grade (in the previous version of the paper their reference score came from the earlier automatic grader) and three with incomplete grades. The `id` column keeps the original numbering, and the code derives from it the row number that identifies each response in stored runs.
- With three grades that differ by at most 1, the rounded mean coincides with the mode.
