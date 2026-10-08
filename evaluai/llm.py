import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, AuthenticationError, NotFoundError, PermissionDeniedError, BadRequestError

_client = None


def get_client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception("Error: Debe definir la variable de entorno OPENAI_API_KEY")
        _client = OpenAI(timeout=60, max_retries=3)
    return _client


# Envía un prompt al modelo, reintentando ante errores
def chat_gpt(text, model, temperature, retries=10, wait=1):
    client = get_client()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                n=1,
                messages=[{"role": "user", "content": text}],
            )
            return [choice.message.content for choice in response.choices]
        except (AuthenticationError, NotFoundError, PermissionDeniedError, BadRequestError):
            # Errores que no se solucionan reintentando (API key, modelo inexistente, etc.)
            raise
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nError en la solicitud: {e}")
                return [""]
            time.sleep(wait * (attempt + 1))


# Envía múltiples prompts en paralelo y retorna las respuestas en el mismo orden
def chat_gpt_multiple(texts, model, temperature, concurrency=50):
    answers = [None] * len(texts)

    def task(idx):
        answers[idx] = chat_gpt(texts[idx], model, temperature)
        print(idx, end="-", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(task, range(len(texts))))  # Propaga la primera excepción fatal
    print()
    return answers


# Extrae diccionario de salida de las respuestas GPT
def extract_dicts(answers_gpt):
    pattern = r'\{[^{}]+\}'

    gpt_dicts = []
    for answer_gpt in answers_gpt:
        try:
            answer = re.findall(pattern, answer_gpt[0])[0]
            try:
                gpt_dicts.append(ast.literal_eval(answer))
            except (ValueError, SyntaxError):
                gpt_dicts.append(json.loads(answer))
        except Exception:
            print(f"Error al extraer diccionario. Respuesta GPT: \n{answer_gpt[0]}\n\n")
            gpt_dicts.append(None)

    return gpt_dicts
