"""
Implementacion de la clase Matrix desde cero.
Sin uso de NumPy ni frameworks externos.

Soporta:
- Creacion de matrices de ceros o con datos
- Transpuesta, producto punto, multiplicacion elemento a elemento
- Suma, resta, multiplicacion escalar
- Norma de Frobenius, copia profunda

Complejidad:
- dot(A, B): O(n * m * p) donde A es n x m y B es m x p
- transpose: O(n * m)
- element_wise: O(n * m)
"""

import math
import random
from typing import List, Optional


class Matrix:
    """
    Matriz de punto flotante implementada como lista de listas.

    Atributos:
        rows (int): Numero de filas.
        cols (int): Numero de columnas.
        data (List[List[float]]): Datos de la matriz.
    """

    def __init__(self, rows: int, cols: int,
                 data: Optional[List[List[float]]] = None):
        """
        Inicializa la matriz.

        Parametros:
        - rows: numero de filas
        - cols: numero de columnas
        - data: lista de listas con los valores (opcional; si None se inicializa en ceros)

        Complejidad: O(rows * cols)
        """
        self.rows = rows
        self.cols = cols

        if data is not None:
            # Hacer copia profunda para evitar aliasing
            self.data = [[float(data[i][j]) for j in range(cols)]
                         for i in range(rows)]
        else:
            self.data = [[0.0] * cols for _ in range(rows)]

    # ----------------------------------------------------------
    # Metodos de fabrica
    # ----------------------------------------------------------

    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Crea una matriz de ceros. O(rows * cols)"""
        return Matrix(rows, cols)

    @staticmethod
    def ones(rows: int, cols: int) -> 'Matrix':
        """Crea una matriz de unos. O(rows * cols)"""
        m = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                m.data[i][j] = 1.0
        return m

    @staticmethod
    def identity(n: int) -> 'Matrix':
        """Crea una matriz identidad n x n. O(n^2)"""
        m = Matrix(n, n)
        for i in range(n):
            m.data[i][i] = 1.0
        return m

    @staticmethod
    def random_normal(rows: int, cols: int,
                      mean: float = 0.0, std: float = 1.0) -> 'Matrix':
        """
        Crea una matriz con valores distribucion normal.
        Usa el metodo Box-Muller.
        Complejidad: O(rows * cols)
        """
        m = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                # Box-Muller transform
                u1 = random.random()
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
                m.data[i][j] = mean + std * z
        return m

    # ----------------------------------------------------------
    # Operaciones basicas
    # ----------------------------------------------------------

    def transpose(self) -> 'Matrix':
        """
        Devuelve la transpuesta de la matriz.
        Complejidad: O(rows * cols)
        """
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def dot(self, other: 'Matrix') -> 'Matrix':
        """
        Producto punto (multiplicacion matricial).
        self debe ser (n x m) y other debe ser (m x p).
        Resultado: matriz (n x p).

        Complejidad: O(n * m * p)

        Lanza ValueError si las dimensiones no son compatibles.
        """
        if self.cols != other.rows:
            raise ValueError(
                f"Dimensiones incompatibles: ({self.rows}x{self.cols}) dot ({other.rows}x{other.cols})"
            )
        n, m, p = self.rows, self.cols, other.cols
        result = Matrix(n, p)
        for i in range(n):
            for k in range(m):
                if self.data[i][k] == 0.0:
                    continue  # Optimizacion menor para matrices dispersas
                for j in range(p):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    def element_wise_mult(self, other: 'Matrix') -> 'Matrix':
        """
        Producto de Hadamard (elemento a elemento).
        Ambas matrices deben tener las mismas dimensiones.
        Complejidad: O(rows * cols)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Dimensiones incompatibles para Hadamard: "
                f"({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
            )
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
        return result

    def scalar_mult(self, scalar: float) -> 'Matrix':
        """
        Multiplicacion por escalar.
        Complejidad: O(rows * cols)
        """
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
        return result

    def add(self, other: 'Matrix') -> 'Matrix':
        """
        Suma elemento a elemento.
        Soporta broadcasting de sesgo: si other es (1 x cols), se suma a cada fila.
        Complejidad: O(rows * cols)
        """
        # Broadcasting: bias (1 x cols) sumado a cada fila
        if other.rows == 1 and self.cols == other.cols:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[0][j]
            return result

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Dimensiones incompatibles para suma: "
                f"({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
            )
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def subtract(self, other: 'Matrix') -> 'Matrix':
        """
        Resta elemento a elemento.
        Complejidad: O(rows * cols)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Dimensiones incompatibles para resta: "
                f"({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
            )
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def add_scalar(self, scalar: float) -> 'Matrix':
        """Suma un escalar a todos los elementos. O(rows * cols)"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + scalar
        return result

    # ----------------------------------------------------------
    # Normas y estadisticas
    # ----------------------------------------------------------

    def frobenius_norm(self) -> float:
        """
        Norma de Frobenius: sqrt(sum(x_ij^2))
        Complejidad: O(rows * cols)
        """
        total = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.data[i][j] ** 2
        return math.sqrt(total)

    def sum(self) -> float:
        """Suma de todos los elementos. O(rows * cols)"""
        total = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.data[i][j]
        return total

    def mean(self) -> float:
        """Media de todos los elementos. O(rows * cols)"""
        return self.sum() / (self.rows * self.cols)

    def max_val(self) -> float:
        """Valor maximo de la matriz. O(rows * cols)"""
        m = self.data[0][0]
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] > m:
                    m = self.data[i][j]
        return m

    def min_val(self) -> float:
        """Valor minimo de la matriz. O(rows * cols)"""
        m = self.data[0][0]
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] < m:
                    m = self.data[i][j]
        return m

    # ----------------------------------------------------------
    # Copia y utilidades
    # ----------------------------------------------------------

    def copy(self) -> 'Matrix':
        """
        Copia profunda de la matriz.
        Complejidad: O(rows * cols)
        """
        return Matrix(self.rows, self.cols, self.data)

    def apply(self, func) -> 'Matrix':
        """
        Aplica una funcion elemento a elemento.
        Complejidad: O(rows * cols * costo_func)
        """
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(self.data[i][j])
        return result

    def clip(self, min_val: float, max_val: float) -> 'Matrix':
        """Recorta los valores al rango [min_val, max_val]. O(rows * cols)"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                v = self.data[i][j]
                result.data[i][j] = max(min_val, min(max_val, v))
        return result

    def row_sum(self) -> 'Matrix':
        """
        Suma las columnas de cada fila, devuelve vector columna (rows x 1).
        Complejidad: O(rows * cols)
        """
        result = Matrix(self.rows, 1)
        for i in range(self.rows):
            s = 0.0
            for j in range(self.cols):
                s += self.data[i][j]
            result.data[i][0] = s
        return result

    def col_mean(self) -> 'Matrix':
        """
        Media de cada columna, devuelve vector fila (1 x cols).
        Complejidad: O(rows * cols)
        """
        result = Matrix(1, self.cols)
        for j in range(self.cols):
            s = 0.0
            for i in range(self.rows):
                s += self.data[i][j]
            result.data[0][j] = s / self.rows
        return result

    def col_std(self) -> 'Matrix':
        """
        Desviacion estandar de cada columna, devuelve vector fila (1 x cols).
        Complejidad: O(rows * cols)
        """
        mean = self.col_mean()
        result = Matrix(1, self.cols)
        for j in range(self.cols):
            s = 0.0
            for i in range(self.rows):
                diff = self.data[i][j] - mean.data[0][j]
                s += diff * diff
            result.data[0][j] = math.sqrt(s / self.rows)
        return result

    # ----------------------------------------------------------
    # Representacion
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"

    def __str__(self) -> str:
        lines = []
        for row in self.data:
            lines.append("  [" + ", ".join(f"{v:8.4f}" for v in row) + "]")
        return f"Matrix {self.rows}x{self.cols}:\n" + "\n".join(lines)
