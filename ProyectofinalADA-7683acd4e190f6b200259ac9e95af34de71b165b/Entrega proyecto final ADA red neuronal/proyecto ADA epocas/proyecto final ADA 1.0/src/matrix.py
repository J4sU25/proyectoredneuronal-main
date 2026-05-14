"""
Módulo de operaciones matriciales - Implementación eficiente sin NumPy
Análisis de complejidad temporal y espacial incluido
"""

import math
import random
from typing import List, Tuple, Optional


class Matrix:
    """
    Clase Matrix: Implementación eficiente de matrices
    Complejidad espacial: O(n*m) donde n=filas, m=columnas
    """
    
    def __init__(self, rows: int, cols: int, data: Optional[List[List[float]]] = None):
        """
        Inicializa una matriz
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        self.rows = rows
        self.cols = cols
        
        if data is None:
            # Inicialización con ceros
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    @staticmethod
    def random(rows: int, cols: int, seed: Optional[int] = None) -> 'Matrix':
        """
        Crea una matriz con valores aleatorios (distribución normal)
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if seed is not None:
            random.seed(seed)
        
        data = [[random.gauss(0, math.sqrt(2.0 / (rows + cols))) 
                 for _ in range(cols)] for _ in range(rows)]
        return Matrix(rows, cols, data)
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        return Matrix(rows, cols)
    
    def copy(self) -> 'Matrix':
        """
        Copia profunda de la matriz
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        new_data = [row[:] for row in self.data]
        return Matrix(self.rows, self.cols, new_data)
    
    def transpose(self) -> 'Matrix':
        """
        Transpone la matriz
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        data = [[self.data[i][j] for i in range(self.rows)] 
                for j in range(self.cols)]
        return Matrix(self.cols, self.rows, data)
    
    def dot(self, other: 'Matrix') -> 'Matrix':
        """
        Producto punto entre matrices: self (m x n) * other (n x p)
        Complejidad temporal: O(m * n * p) - Método directo
        Complejidad espacial: O(m * p)
        """
        if self.cols != other.rows:
            raise ValueError(f"Dimensiones incompatibles: ({self.rows}x{self.cols}) * ({other.rows}x{other.cols})")
        
        result = Matrix.zeros(self.rows, other.cols)
        
        # Implementación estándar O(m*n*p)
        for i in range(self.rows):
            for k in range(self.cols):
                val = self.data[i][k]
                if val != 0:  # Optimización: saltar ceros
                    for j in range(other.cols):
                        result.data[i][j] += val * other.data[k][j]
        
        return result
    
    def element_wise_mult(self, other: 'Matrix') -> 'Matrix':
        """
        Multiplicación elemento a elemento (Hadamard)
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
        
        return result
    
    def add(self, other: 'Matrix') -> 'Matrix':
        """
        Suma elemento a elemento
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        
        return result
    
    def subtract(self, other: 'Matrix') -> 'Matrix':
        """
        Resta elemento a elemento
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        
        return result
    
    def scalar_mult(self, scalar: float) -> 'Matrix':
        """
        Multiplicación por escalar
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(rows * cols)
        """
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
        
        return result
    
    def apply_function(self, func) -> 'Matrix':
        """
        Aplica una función a cada elemento
        Complejidad temporal: O(rows * cols * f) donde f es costo de func
        Complejidad espacial: O(rows * cols)
        """
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(self.data[i][j])
        
        return result
    
    def sum(self) -> float:
        """
        Suma todos los elementos
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(1)
        """
        total = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.data[i][j]
        
        return total
    
    def mean(self) -> float:
        """
        Media de todos los elementos
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(1)
        """
        return self.sum() / (self.rows * self.cols)
    
    def frobenius_norm(self) -> float:
        """
        Norma de Frobenius: sqrt(suma de cuadrados)
        Complejidad temporal: O(rows * cols)
        Complejidad espacial: O(1)
        """
        sum_squares = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                sum_squares += val * val
        
        return math.sqrt(sum_squares)
