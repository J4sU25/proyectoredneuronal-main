"""
Funciones de activacion implementadas desde cero.
Sin uso de NumPy ni frameworks externos.

Incluye:
- ReLU y su derivada
- Sigmoid y su derivada
- Tanh y su derivada
- Lineal (identidad)
- Softmax y su derivada (para clasificacion multiclase)
- apply / apply_derivative para uso en capas del MLP

Complejidades:
- apply_element: O(1) por elemento
- softmax (batch): O(batch_size * num_classes)
"""

import math
from typing import List
from src.matrix import Matrix


class ActivationFunction:
    """
    Clase estatica con implementacion de todas las funciones de activacion.
    """

    # ----------------------------------------------------------
    # ReLU - Rectified Linear Unit
    # f(x) = max(0, x)
    # Complejidad: O(1)
    # ----------------------------------------------------------

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def relu_derivative(x: float) -> float:
        """f'(x) = 1 si x > 0, 0 si x <= 0"""
        return 1.0 if x > 0.0 else 0.0

    # ----------------------------------------------------------
    # Sigmoid
    # f(x) = 1 / (1 + e^(-x))
    # Complejidad: O(1)
    # ----------------------------------------------------------

    @staticmethod
    def sigmoid(x: float) -> float:
        # Numericamente estable: evitar overflow
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            return ex / (1.0 + ex)

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        """f'(x) = sigmoid(x) * (1 - sigmoid(x))"""
        s = ActivationFunction.sigmoid(x)
        return s * (1.0 - s)

    # ----------------------------------------------------------
    # Tanh - Tangente hiperbolica
    # f(x) = (e^x - e^-x) / (e^x + e^-x)
    # Complejidad: O(1)
    # ----------------------------------------------------------

    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x: float) -> float:
        """f'(x) = 1 - tanh(x)^2"""
        t = math.tanh(x)
        return 1.0 - t * t

    # ----------------------------------------------------------
    # Lineal (identidad)
    # f(x) = x
    # Se usa tipicamente en la ultima capa antes del Softmax
    # Complejidad: O(1)
    # ----------------------------------------------------------

    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def linear_derivative(x: float) -> float:
        """f'(x) = 1"""
        return 1.0

    # ----------------------------------------------------------
    # Aplicar a matrices completas
    # ----------------------------------------------------------

    @staticmethod
    def apply(matrix: Matrix, activation: str) -> Matrix:
        """
        Aplica la funcion de activacion a cada elemento de la matriz.

        Parametros:
        - matrix: entrada
        - activation: 'relu', 'sigmoid', 'tanh', 'linear'

        Complejidad: O(rows * cols)
        """
        activation = activation.lower()

        if activation == 'relu':
            func = ActivationFunction.relu
        elif activation == 'sigmoid':
            func = ActivationFunction.sigmoid
        elif activation == 'tanh':
            func = ActivationFunction.tanh
        elif activation in ('linear', 'identity', 'none'):
            func = ActivationFunction.linear
        else:
            raise ValueError(f"Funcion de activacion desconocida: '{activation}'")

        return matrix.apply(func)

    @staticmethod
    def apply_derivative(matrix: Matrix, activation: str) -> Matrix:
        """
        Aplica la derivada de la funcion de activacion elemento a elemento.
        Se usa durante el backward pass.

        Parametros:
        - matrix: salida pre-activacion (Z = XW + b) o la activacion misma
        - activation: 'relu', 'sigmoid', 'tanh', 'linear'

        Complejidad: O(rows * cols)
        """
        activation = activation.lower()

        if activation == 'relu':
            func = ActivationFunction.relu_derivative
        elif activation == 'sigmoid':
            func = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            func = ActivationFunction.tanh_derivative
        elif activation in ('linear', 'identity', 'none'):
            func = ActivationFunction.linear_derivative
        else:
            raise ValueError(f"Funcion de activacion desconocida: '{activation}'")

        return matrix.apply(func)


# ----------------------------------------------------------
# Softmax
# Convierte logits en probabilidades (suma = 1 por fila)
# Complejidad: O(batch_size * num_classes)
# ----------------------------------------------------------

def softmax(matrix: Matrix) -> Matrix:
    """
    Softmax estable numericamente aplicado por filas.

    Para cada fila x:
        1. Restar el maximo (estabilidad numerica)
        2. exp(x_i - max)
        3. Dividir por la suma

    Complejidad: O(rows * cols)
    """
    result = Matrix(matrix.rows, matrix.cols)
    for i in range(matrix.rows):
        row = matrix.data[i]
        # Estabilidad numerica: restar el maximo
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps)
        if total == 0.0:
            total = 1e-10  # Evitar division por cero
        result.data[i] = [e / total for e in exps]
    return result


def softmax_derivative(predictions: Matrix, targets: Matrix) -> Matrix:
    """
    Derivada de la funcion de perdida cross-entropy + softmax combinada.

    Cuando la ultima capa usa activacion lineal seguida de softmax y
    la perdida es cross-entropy, el gradiente combinado es simplemente:
        dL/dZ_last = predictions - targets

    Esto es la formula analitica exacta que simplifica el calculo.

    Parametros:
    - predictions: salidas del forward pass despues de softmax (batch x clases)
    - targets: etiquetas one-hot verdaderas (batch x clases)

    Complejidad: O(batch_size * num_classes)
    """
    if predictions.rows != targets.rows or predictions.cols != targets.cols:
        raise ValueError(
            f"Dimensiones incompatibles: predictions {predictions.rows}x{predictions.cols}"
            f" vs targets {targets.rows}x{targets.cols}"
        )
    return predictions.subtract(targets)
