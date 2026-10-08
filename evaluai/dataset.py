import random

import pandas as pd

MANDATORY_COLS = ["context", "question", "answer", "real_eval", "dataset"]


# Muestra información relevante del dataset
def show_dataset_info(dataset):
    print(dataset.head().to_string())
    print()
    print(dataset.value_counts("real_eval"), end="\n\n")
    print(dataset.value_counts("dataset"))


# Carga un dataset a partir de un archivo xlsx y valida sus columnas
def load_dataset(path, sheet_name, column_data, verbose=True):
    df = pd.read_excel(path, sheet_name=sheet_name)

    for key in MANDATORY_COLS:
        if key not in column_data.keys():
            raise Exception(f"Error: Debe especificar la columna para la variable {key}")

        value = column_data[key]
        if value not in df.columns:
            raise Exception(f"Error: La columna {value} no existe. Columnas disponibles: {list(df.columns)}")

        df = df.rename(columns={value: key})

    df = df[MANDATORY_COLS]
    df['row'] = df.index + 2
    if verbose: show_dataset_info(df)
    return df


# Retorna un conjunto de datos de entrenamiento o prueba
def generate_set(dataset, set_size, seed=42, balance=False, exclude_set=None):
    random.seed(seed)

    group_size = set_size // 4  # Tamaño grupo para set balanceado
    proportions = dataset['real_eval'].value_counts(normalize=True)  # Para set no balanceado
    samples_per_class = (proportions * set_size).round().astype(int)

    # Crear set excluyendo las filas de exclude_set
    final_set = dataset
    if exclude_set is not None:
        final_set = dataset[~dataset['row'].isin(exclude_set['row'])]

    columns = dataset.columns.tolist()
    if balance:
        final_set = final_set.groupby('real_eval', group_keys=False)[columns].apply(
            lambda x: x.sample(group_size, random_state=random.randint(0, 100000))
        )
    else:
        final_set = final_set.groupby('real_eval', group_keys=False)[columns].apply(
            lambda x: x.sample(samples_per_class[x.name], random_state=random.randint(0, 100000))
        )

    return final_set
