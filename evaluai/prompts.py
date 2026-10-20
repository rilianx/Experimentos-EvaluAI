import copy
import os
import re

DEFAULT_CRITERION_DESC = "integer between 0 and 10"


class Prompt():
    def __init__(self, structure, instructions, base_folder):
        self.structure = structure
        self.instructions = instructions
        self.base_folder = base_folder
        self.raw_text_structure = None
        self.text_structure = None
        self.criteria = None
        self.output_instructions = None
        self.prompt = None

        self.read_files()
        self.extract_metadata()
        self.build_prompt()

    # Retorna la estructura base del prompt (diccionario)
    def base_structure(self):
        structure = copy.deepcopy(self.structure)
        structure['instructions'] = {}
        for i in self.instructions:
            structure['instructions'][i] = structure[i]
            structure.pop(i, None)
        return structure

    # Crea un diccionario con el contenido de cada archivo en la estructura
    def read_files(self):
        self.raw_text_structure = copy.deepcopy(self.structure)

        for key, value in self.raw_text_structure.items():
            if key == "instructions": continue

            path = f"{self.base_folder}/{key}/{value}"
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.raw_text_structure[key] = f.read() + "\n\n"
            except OSError:
                raise Exception(f"Error: El archivo {path} no existe")

    # Extrae metadatos de los archivos como los criterios e instrucciones de salida.
    # Formatos aceptados para la primera línea:
    #   "#clave: descripción"  -> clave de salida con su descripción
    #   "#descripción"         -> se usa el nombre de la carpeta como clave
    # En el archivo de score, cada línea "$criterio[: descripción]" define un criterio.
    # Si el archivo de score no define criterios, se usa "score" como único criterio.
    def extract_metadata(self):
        self.text_structure = copy.deepcopy(self.raw_text_structure)
        self.output_instructions = {}
        self.criteria = []

        for key in self.text_structure:
            value = self.text_structure[key]

            if value.startswith('#') and not value.startswith('##'):
                first_line, _, rest = value.partition('\n')
                m = first_line[1:].strip()
                if ":" in m:
                    mkey, mvalue = m.split(":", 1)
                else:
                    mkey, mvalue = key, m
                self.output_instructions[mkey.strip()] = mvalue.strip()
                self.text_structure[key] = rest

            if key == 'score':
                lines = self.text_structure['score'].split('\n')
                kept = []
                for line in lines:
                    if line.startswith('$'):
                        m = line[1:].strip()
                        if ":" in m:
                            mkey, mvalue = m.split(":", 1)
                        else:
                            mkey, mvalue = m, DEFAULT_CRITERION_DESC
                        self.criteria.append(mkey.strip())
                        self.output_instructions[mkey.strip()] = mvalue.strip()
                    else:
                        kept.append(line)
                self.text_structure['score'] = '\n'.join(kept)

                if not self.criteria:
                    self.criteria.append('score')
                    self.output_instructions['score'] = DEFAULT_CRITERION_DESC

    # Construye el prompt en formato string
    def build_prompt(self):
        self.prompt = ""
        for key, value in self.text_structure.items():
            self.prompt += value

        output = self.build_output()
        self.prompt += output

    def build_output(self):
        output = "I expect a dict in python as answer: {{"
        output += ', '.join(f'"{key}": \'{value}\'' for key, value in self.output_instructions.items())

        output += "}}\n\nPython dict:"
        return output


# Procesa y elimina los diccionarios anidados de prompt_data
def normalize_prompt_dict(prompt_data):
    prompt_data = copy.deepcopy(prompt_data)
    instructions = []
    after_instructions = {}
    found_target = False

    if isinstance(prompt_data.get("instructions"), dict):
        for key, value in prompt_data.items():
            if found_target:
                after_instructions[key] = value
            if key == "instructions":
                found_target = True

        for (key, value) in prompt_data["instructions"].items():
            prompt_data[key] = value
            instructions.append(key)

        for key, value in after_instructions.items():
            del prompt_data[key]
            prompt_data[key] = value

    prompt_data["instructions"] = "Instructions:\n"
    return prompt_data, instructions


# Retorna la lista de archivos para reemplazar el comodín *
def expand_prompt_data(prompt_data, prompt_folder):
    wildcard_field = None
    for key, value in prompt_data.items():
        if value == "*":
            wildcard_field = key
            break

    if not wildcard_field: return None, None

    wildcard_files = []
    path = f"{prompt_folder}/{wildcard_field}"
    for file in sorted(os.listdir(path)):
        if os.path.isfile(os.path.join(path, file)):
            wildcard_files.append(file)

    return wildcard_field, wildcard_files


# Genera una lista con los prompts a evaluar
def generate_prompts(prompt_data, prompt_folder, visualize=True):
    template, instructions = normalize_prompt_dict(prompt_data)
    wildcard_field, wildcard_files = expand_prompt_data(template, prompt_folder)
    prompts = []

    if wildcard_field is None:
        prompt = Prompt(template, instructions, prompt_folder)
        prompt.label = "prompt"
        prompts.append(prompt)

        if visualize: print(prompt.prompt)
        return prompts

    for file in wildcard_files:
        structure = copy.deepcopy(template)
        structure[wildcard_field] = file
        prompt = Prompt(structure, instructions, prompt_folder)
        prompt.label = os.path.splitext(file)[0]
        prompts.append(prompt)

    # Visualizar
    if visualize:
        template = copy.deepcopy(prompts[0])
        template.raw_text_structure[wildcard_field] = f"{{{wildcard_field}}}\n\n"
        template.extract_metadata()
        template.build_prompt()
        print(template.prompt)

        print(f"\n\nArchivos a utilizar ({len(wildcard_files)}):\n")
        print("\n".join(wildcard_files))

    return prompts
