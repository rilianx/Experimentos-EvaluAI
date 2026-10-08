# Framework para experimentos EvaluAI

El objetivo de este framework consiste en proporcionar un conjunto de funciones predefinidas para facilitar la evaluación y comparación de prompts en el contexto del proyecto EvaluAI.

## Instalación

Requiere Python 3.9 o superior.

```bash
pip install -r requirements.txt
cp .env.example .env              # y escribir la API key de OpenAI
cp config.example.json config.json
```

La API key se lee de la variable de entorno **OPENAI_API_KEY** (o del archivo `.env`).

Los *mini-prompts* se obtienen del repositorio [GPTEvaluator](https://github.com/rilianx/GPTEvaluator), que se clona automáticamente en `GPTEvaluator/` la primera vez que se ejecuta el script.

El dataset usado en los experimentos (`datasets_v2.xlsx`) está en [Google Drive](https://docs.google.com/spreadsheets/d/1rlM1Z31t6qbMVL5HK7-5g1cFmlcAC7ei/edit?gid=803566578#gid=803566578). Se debe descargar como *xlsx* y dejar en la raíz del repositorio (o indicar su ruta en `config.json`).

## Uso rápido

Todo el experimento se configura en `config.json` (ver `config.example.json`), y se ejecuta con `main.py`:

```bash
python main.py prompts                          # Muestra el o los prompts que se van a evaluar
python main.py generate                         # Genera las respuestas del modelo en Responses/
python main.py evaluate --prefix feedback_normal  # Evalúa los archivos de Responses/ que empiezan con el prefijo
python main.py read --prefix T_ --normalize     # Muestra las métricas de los archivos de Evals/
python main.py all                              # generate + evaluate sobre las respuestas recién generadas
```

Cada subcomando acepta `--config <archivo>` (por defecto `config.json`); `evaluate` y `read` también aceptan una lista explícita de archivos.

Las funciones también se pueden usar directamente desde Python:

```python
from evaluai import load_dataset, generate_prompts, generate_responses, evaluate_mul, read_evals
```

El notebook original `Framework_EvaluAI.ipynb` se mantiene como referencia; el código vigente está en el paquete `evaluai/`.

Las secciones siguientes describen cada parte de la configuración y las funciones equivalentes.

## Cargar dataset

Se debe usar un archivo *xlsx* con al menos las siguientes columnas:
* **Contexto**: Conocimiento previo que necesita el modelo para evaluar la respuesta del estudiante.
* **Pregunta**: Pregunta evaluada.
* **Respuesta**: Respuesta del estudiante.
* **Evaluación manual**: Puntaje de referencia dado por uno o más evaluadores humanos.
* **Dataset de origen**: Dataset del cual provienen los datos para una fila en particular. Podría hacer referencia a diferentes evaluaciones, categorías de alumnos, cursos, etc.

El nombre de las columnas se debe indicar (clave `dataset.columns` de `config.json`) en un diccionario con la siguiente estructura:

```
column_data = {
    "context": ...,
    "question": ...,
    "answer": ...,
    "real_eval": ...,
    "dataset": ...
}
```

Por último, se debe cargar el dataset con la función `load_dataset(path, sheet_name, column_data)`

- **path** - Ruta del archivo *xlsx* a utilizar.
- **sheet_name** - Nombre de la hoja donde se encuentran los datos.
- **column_data** - Diccionario con el nombre de las columnas.

## Definir prompts a utilizar

Los prompts se componen de varios *mini-prompts*. Cada *mini-prompt* representa una parte del prompt completo, como la pregunta, la respuesta del estudiante, la instrucción para solicitar el puntaje, etc.

El framework cuenta con algunos *mini-prompts* predefinidos en la ruta *GPTEvaluator/Experiments/Miniprompts_v2*, aunque también se pueden modificar o agregar nuevos *mini-prompts* en caso de ser necesario.

El prompt a evaluar se debe construir mediante un diccionario, donde la clave corresponde al nombre del *mini-prompt* o componente (question, examples, feedback...) y el valor corresponde a la variante a utilizar (feedback_minimal, feedback_full, feedback_normal...). Se debe respetar la estructura de archivos, donde los componentes son las carpetas y las variantes son archivos *txt* dentro de cada carpeta.

Usar como guía el siguiente ejemplo:

```
prompt_data = {
    "examples_basic": "examples_XXXX_basic.txt",
    "context": "...",
    "question": "...",
    "answer": "...",
    "instructions": {
        "analysis": "...",
        "feedback": "...",
        "score": "...",
    }
}
```

Además, se puede utilizar el comodín * como valor para probar con todos los miniprompts de una carpeta en particular.

Finalmente, cargar el o los prompts con la función `generate_prompts(prompt_data, prompt_folder, visualize=True)`

- **prompt_data** - Diccionario con la estructura del prompt.
- **prompt_folder** - Ruta de la carpeta donde se encuentra la colección de miniprompts.
- **visualize** - Para imprimir el prompt generado.

En `config.json` este diccionario corresponde a la clave `prompt`, y la carpeta a `prompt_folder`.

Formato de metadatos de los *mini-prompts*:
- Una primera línea `#clave: descripción` agrega `clave` al diccionario de salida pedido al modelo. Si no tiene `:` (por ejemplo `#feedback`), se usa el nombre de la carpeta como clave.
- En la carpeta `score`, cada línea `$criterio` o `$criterio: descripción` define un criterio a puntuar. Si el archivo no define criterios, se usa `score` como único criterio.

## Ejecutar el experimento

El experimento se divide en dos etapas: generar las respuestas del modelo y luego evaluarlas.

### 1. Generar respuestas (`python main.py generate`)

Equivale a `generate_responses(dataset, prompts, repetitions, set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1, balance_set=False, repeat_set=False)` y se configura en la clave `generate` de `config.json` (junto con `model` y `temperature`).

- **repetitions** - Número de conjuntos de prueba a evaluar para cada prompt.
- **set_size** - Tamaño de cada conjunto de prueba.
- **seed** - Semilla utilizada para la generación de los conjuntos.
- **model** - Nombre del modelo a utilizar, por ejemplo `gpt-4o-mini`.
- **temperature** - Temperatura del modelo.
- **balance_set** - Para que cada conjunto tenga la misma cantidad de muestras para cada puntaje.
- **repeat_set** - Para mantener el mismo conjunto en todas las repeticiones.

Se genera un archivo `Responses/<variante>_<fecha>.xlsx` por prompt, donde `<variante>` es el archivo usado en el comodín * (o `prompt` si no se usó).

### 2. Evaluar respuestas (`python main.py evaluate`)

Equivale a `evaluate_mul(filenames, dataset, eval_function, eval_params=None, train_set_size=100, seed=42, model="gpt-4o-mini", temperature=0.1)` y se configura en la clave `evaluate`.

- **eval_function** - Método para convertir los puntajes del modelo (escala 0-10) a escala 0-3. Puede ser `map2`, `map2-simple`, `map2-mini`, `map4` o `map4-mini`.
- **eval_params** - Lista de parámetros usados para la conversión de puntajes. Dependen de la `eval_function` elegida. Si es `null`, los parámetros se ajustan automáticamente con un conjunto de entrenamiento (lo que implica nuevas consultas al modelo).
- **train_set_size** - Tamaño del conjunto de entrenamiento.
- **seed** - Semilla utilizada para generar los conjuntos de entrenamiento.

Los resultados se guardan en `Evals/` (con prefijo `T_` cuando los parámetros fueron ajustados) y se muestran dos tablas con la media y desviación estándar de cada métrica.

### 3. Revisar resultados (`python main.py read`)

Equivale a `read_evals(filenames, normalize=False)`.

- **filenames** - Archivos de evaluación a leer (por defecto, los de `Evals/` que empiezan con `--prefix`).
- **normalize** - Para indicar si las métricas MSE y MAE deberían normalizarse entre 0 y 1.

Cuando se lee un único archivo, y este fue generado con `repeat_set` o con una sola repetición, se guarda además un gráfico interactivo con la distribución de puntajes en `Plots/` (con `--show` se abre en el navegador).

![Ejemplo de gráfico de distribución de puntajes](/images/plot.png)
