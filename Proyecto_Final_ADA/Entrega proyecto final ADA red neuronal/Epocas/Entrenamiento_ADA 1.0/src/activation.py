"""
Funciones de activación y sus derivadas
Análisis de complejidad: O(1) por elemento en matrices
"""

import math
from src.matrix import Matrix


class ActivationFunction:
    """Clase base para funciones de activación"""
    
    @staticmethod
    def relu(x: float) -> float:
        """
        ReLU: max(0, x)
        Complejidad: O(1)
        """
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        """
        Derivada de ReLU: 1 si x > 0, 0 si x <= 0
        Complejidad: O(1)
        """
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Sigmoid: 1 / (1 + e^-x)
        Complejidad: O(1)
        """
        if x > 100:  # Evitar overflow
            return 1.0
        if x < -100:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        """
        Derivada de sigmoid: sigmoid(x) * (1 - sigmoid(x))
        Complejidad: O(1)
        """
        s = ActivationFunction.sigmoid(x)
        return s * (1.0 - s)
    
    @staticmethod
    def tanh(x: float) -> float:
        """
        Tanh: (e^x - e^-x) / (e^x + e^-x)
        Complejidad: O(1)
        """
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        """
        Derivada de tanh: 1 - tanh²(x)
        Complejidad: O(1)
        """
        t = math.tanh(x)
        return 1.0 - t * t
    
    @staticmethod
    def apply_activation(matrix: Matrix, activation: str) -> Matrix:
        """
        Aplica función de activación a toda una matriz
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if activation == 'relu':
            return matrix.apply_function(ActivationFunction.relu)
        elif activation == 'sigmoid':
            return matrix.apply_function(ActivationFunction.sigmoid)
        elif activation == 'tanh':
            return matrix.apply_function(ActivationFunction.tanh)
        elif activation == 'linear':
            return matrix
        else:
            raise ValueError(f"Función de activación desconocida: {activation}")
    
    @staticmethod
    def apply_derivative(matrix: Matrix, activation: str) -> Matrix:
        """
        Aplica derivada de función de activación a toda una matriz
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if activation == 'relu':
            return matrix.apply_function(ActivationFunction.relu_derivative)
        elif activation == 'sigmoid':
            return matrix.apply_function(ActivationFunction.sigmoid_derivative)
        elif activation == 'tanh':
            return matrix.apply_function(ActivationFunction.tanh_derivative)
        elif activation == 'linear':
            return Matrix(matrix.rows, matrix.cols, 
                         [[1.0 for _ in range(matrix.cols)] for _ in range(matrix.rows)])
        else:
            raise ValueError(f"Función de activación desconocida: {activation}")


def softmax(logits: Matrix) -> Matrix:
    """
    Softmax normalizado para estabilidad numérica
    logits: matriz (batch_size, num_classes)
    
    Complejidad temporal: O(batch_size * num_classes)
    Complejidad espacial: O(batch_size * num_classes)
    """
    result = logits.copy()
    
    for i in range(result.rows):
        # Encontrar máximo para estabilidad numérica
        max_val = max(result.data[i])
        
        # Restar máximo y calcular exp
        exp_sum = 0.0
        for j in range(result.cols):
            result.data[i][j] = math.exp(result.data[i][j] - max_val)
            exp_sum += result.data[i][j]
        
        # Normalizar
        for j in range(result.cols):
            result.data[i][j] /= exp_sum
    
    return result


def softmax_derivative(softmax_output: Matrix, target: Matrix) -> Matrix:
    """
    Derivada de cross-entropy con softmax
    Complejidad temporal: O(batch_size * num_classes)
    Complejidad espacial: O(batch_size * num_classes)
    """
    return softmax_output.subtract(target)
