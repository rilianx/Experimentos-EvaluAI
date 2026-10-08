from scipy.optimize import differential_evolution
from abc import ABC, abstractmethod

import numpy as np

class ScoreWeighter():
    @staticmethod
    def eval(x, theta, right_offset):
        return np.dot(x, theta[:-right_offset])

class MapOptimizer(ABC):
    def __init__(self, map_params_size):
        self.map_params_size = map_params_size

    def optimize(self, criteria_scores, real_scores):
        bounds =  [(0, 1) for _ in range(len(criteria_scores[0]))] + [(0, 10)] * self.map_params_size
        result = differential_evolution(self.error, bounds, args=(criteria_scores, real_scores), seed=1, strategy='rand1exp', mutation=(0,1), recombination=1)
        return result.x.tolist()

    def error(self, theta, x, y):
        y_pred = self.f(x, theta)
        mse = np.sum((y - y_pred) ** 2)

        # Penalización cuando suma de ponderaciones != 1
        weights = theta[:-self.map_params_size]
        penalty = 1e6 * np.abs(np.sum(weights) - 1)

        # Penalización cuando parámetros de mapeo no están de menor a mayor
        map_params = theta[-self.map_params_size:]
        penalty += sum((map_params[i] - map_params[i+1]) * 1e5 for i in range(len(map_params)-1) if map_params[i] > map_params[i+1])
        return mse + penalty

    @abstractmethod
    def f(self, x, theta):
        pass

class MapOptimizer4(MapOptimizer):
    def __init__(self):
        super().__init__(4)

    def map_array(self, w_scores, theta):
        a, b, c, d = theta[-4:]
        def map(x):
            if x <= a:
                return 0
            if a < x <= b:
                return (x - a) / (b - a)
            if b < x <= c:
                return 1 + (x - b) / (c - b)
            if c < x <= d:
                return 2 + (x - c) / (d - c)
            else:
                return 3
        return np.array([map(x) for x in w_scores])

    def f(self, x, theta):
        w_scores = ScoreWeighter.eval(x, theta, 4)
        return self.map_array(w_scores, theta)

class MapOptimizer2(MapOptimizer):
    def __init__(self):
        super().__init__(2)

    def map_array(self, w_scores, theta):
        a, b = theta[-2:]
        def map(x):
            if x <= a:
                return x / a
            if a < x <= b:
                return 1 + (x - a) / (b - a)
            else:
                return 2 + (x - b) / (10 - b)
        return np.array([map(x) for x in w_scores])

    def f(self, x, theta):
        w_scores = ScoreWeighter.eval(x, theta, 2)
        return self.map_array(w_scores, theta)

class MapOptimizer2Simple(MapOptimizer):
    def __init__(self):
        super().__init__(2)

    def map_array(self, w_scores, theta, eps=1e-9):
        a, b = theta[-2:]
        def map(x):
            if x <= a:
                return x*eps
            if a < x <= b:
                return 3 * (x - a) / (b - a)
            else:
                return 3 + x*eps
        return np.array([map(x) for x in w_scores])

    def f(self, x, theta):
        w_scores = ScoreWeighter.eval(x, theta, 2)
        return self.map_array(w_scores, theta)

class MapOptimizer2Mini(MapOptimizer):
    def __init__(self):
        super().__init__(1)

    def map_array(self, w_scores, theta):
        a = theta[-1]
        def map(x):
            b = 10 - a
            if x <= a:
                return 0
            if a < x <= b:
                return 3 * (x - a) / (b - a)
            else:
                return 3
        return np.array([map(x) for x in w_scores])

    def f(self, x, theta):
        w_scores = ScoreWeighter.eval(x, theta, 1)
        return self.map_array(w_scores, theta)

class MapOptimizer4Mini(MapOptimizer):
    def __init__(self):
        super().__init__(3)

    def map_array(self, w_scores, theta):
        a, b, c = theta[-3:]
        def map(x):
            d = 10 - a
            if x <= a:
                return 0
            if a < x <= b:
                return (x - a) / (b - a)
            if b < x <= c:
                return 1 + (x - b) / (c - b)
            if c < x <= d:
                return 2 + (x - c) / (d - c)
            else:
                return 3
        return np.array([map(x) for x in w_scores])

    def f(self, x, theta):
        w_scores = ScoreWeighter.eval(x, theta, 3)
        return self.map_array(w_scores, theta)

EVAL_FUNCTIONS = ["map2", "map2-simple", "map2-mini", "map4", "map4-mini", "map"]

# Obtiene los parámetros óptimos para disminuir el error
def optimize_params(criteria_scores, real_scores, eval_function):
    if eval_function == "map4":
        params = MapOptimizer4().optimize(criteria_scores, real_scores)
    if eval_function == "map4-mini":
        params = MapOptimizer4Mini().optimize(criteria_scores, real_scores)
    if eval_function == "map2" or eval_function == "map":
        params = MapOptimizer2().optimize(criteria_scores, real_scores)
    if eval_function == "map2-simple":
        params = MapOptimizer2Simple().optimize(criteria_scores, real_scores)
    if eval_function == "map2-mini":
        params = MapOptimizer2Mini().optimize(criteria_scores, real_scores)
    if eval_function not in EVAL_FUNCTIONS:
        raise ValueError(f"Error: eval_function debe ser una de {EVAL_FUNCTIONS}")

    return params

# Convierte los puntajes del modelo (0-10) a la escala 0-3 usando los parámetros dados
def convert_gpt_scores(criteria_scores, real_scores, eval_function, eval_params):
    if eval_function == "map4":
        return MapOptimizer4().f(criteria_scores, eval_params)
    if eval_function == "map4-mini":
        return MapOptimizer4Mini().f(criteria_scores, eval_params)
    if eval_function == "map2" or eval_function == "map":
        return MapOptimizer2().f(criteria_scores, eval_params)
    if eval_function == "map2-simple":
        return MapOptimizer2Simple().f(criteria_scores, eval_params)
    if eval_function == "map2-mini":
        return MapOptimizer2Mini().f(criteria_scores, eval_params)
    raise ValueError(f"Error: eval_function debe ser una de {EVAL_FUNCTIONS}")
