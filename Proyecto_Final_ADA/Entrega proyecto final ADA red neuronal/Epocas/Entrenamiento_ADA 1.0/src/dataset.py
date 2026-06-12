"""
Generador de Datasets Sinteticos para clasificacion.
Implementado desde cero sin NumPy ni frameworks externos.

Genera:
- Datasets de clasificacion sintetica (clusters gaussianos)
- Datasets tipo MNIST (imagenes aleatorias de tamano variable)
- Normalizacion Z-score por caracteristica
- Division train/val con mezcla aleatoria

Complejidad:
- generate_synthetic_classification: O(N * F) donde N=muestras, F=features
- normalize_dataset: O(N * F)
- split_dataset: O(N)
"""

import math
import random
from typing import Tuple, List, Optional

from src.matrix import Matrix


class DatasetGenerator:
    """
    Clase estatica para generar y procesar datasets.
    """

    # ----------------------------------------------------------
    # Generador de clasificacion sintetica
    # ----------------------------------------------------------

    @staticmethod
    def generate_synthetic_classification(
            num_samples: int,
            num_features: int,
            num_classes: int,
            seed: Optional[int] = None) -> Tuple[Matrix, Matrix]:
        """
        Genera un dataset de clasificacion sintetica con clusters gaussianos.

        Para cada clase se genera un centroide aleatorio y las muestras
        se distribuyen alrededor de el con ruido gaussiano.

        Parametros:
        - num_samples: numero total de muestras
        - num_features: numero de caracteristicas (dimensiones)
        - num_classes: numero de clases
        - seed: semilla para reproducibilidad (None = aleatorio)

        Retorna:
        - X: Matrix (num_samples x num_features), features
        - y: Matrix (num_samples x num_classes), etiquetas one-hot

        Complejidad: O(num_samples * num_features)
        """
        if seed is not None:
            random.seed(seed)

        # Generar centroides para cada clase
        centroids = []
        for c in range(num_classes):
            centroid = [random.uniform(-3.0, 3.0) for _ in range(num_features)]
            centroids.append(centroid)

        # Generar muestras
        X_data = []
        y_data = []

        samples_per_class = num_samples // num_classes
        remaining = num_samples - samples_per_class * num_classes

        for c in range(num_classes):
            count = samples_per_class + (1 if c < remaining else 0)
            for _ in range(count):
                # Muestra = centroide + ruido gaussiano
                sample = []
                for f in range(num_features):
                    noise = random.gauss(0.0, 1.0)
                    sample.append(centroids[c][f] + noise)
                X_data.append(sample)

                # Etiqueta one-hot
                label = [0.0] * num_classes
                label[c] = 1.0
                y_data.append(label)

        # Barajar el dataset
        combined = list(zip(X_data, y_data))
        random.shuffle(combined)
        X_data, y_data = zip(*combined) if combined else ([], [])

        X = Matrix(num_samples, num_features, list(X_data))
        y = Matrix(num_samples, num_classes, list(y_data))

        return X, y

    # ----------------------------------------------------------
    # Generador tipo MNIST
    # ----------------------------------------------------------

    @staticmethod
    def generate_mnist_like(
            num_samples: int,
            num_classes: int,
            img_size: int = 8,
            seed: Optional[int] = None) -> Tuple[Matrix, Matrix]:
        """
        Genera un dataset sintetico que simula imagenes tipo MNIST.

        Cada imagen es de tamano img_size x img_size = img_size^2 features.
        Los valores son pixeles en [0, 1].

        Parametros:
        - num_samples: numero de muestras
        - num_classes: numero de clases (digitos 0..num_classes-1)
        - img_size: tamano de la imagen cuadrada (img_size x img_size)
        - seed: semilla para reproducibilidad

        Retorna:
        - X: Matrix (num_samples x img_size^2)
        - y: Matrix (num_samples x num_classes), one-hot

        Complejidad: O(num_samples * img_size^2)
        """
        if seed is not None:
            random.seed(seed)

        num_features = img_size * img_size

        # Crear patrones base para cada clase (imagenes de referencia)
        patterns = []
        for c in range(num_classes):
            # Cada clase tiene un patron base diferente
            pattern = [random.uniform(0.0, 1.0) for _ in range(num_features)]
            patterns.append(pattern)

        X_data = []
        y_data = []

        for i in range(num_samples):
            clase = i % num_classes
            # Muestra = patron + ruido pequeno
            sample = []
            for f in range(num_features):
                noise = random.gauss(0.0, 0.15)
                val = patterns[clase][f] + noise
                val = max(0.0, min(1.0, val))  # Clip [0, 1]
                sample.append(val)
            X_data.append(sample)

            label = [0.0] * num_classes
            label[clase] = 1.0
            y_data.append(label)

        # Barajar
        combined = list(zip(X_data, y_data))
        random.shuffle(combined)
        X_data, y_data = zip(*combined) if combined else ([], [])

        X = Matrix(num_samples, num_features, list(X_data))
        y = Matrix(num_samples, num_classes, list(y_data))

        return X, y

    # ----------------------------------------------------------
    # Normalizacion Z-score
    # ----------------------------------------------------------

    @staticmethod
    def normalize_dataset(X: Matrix,
                           epsilon: float = 1e-8) -> Matrix:
        """
        Normaliza el dataset usando Z-score por caracteristica.

        Para cada columna j:
            X_norm[:, j] = (X[:, j] - mean_j) / (std_j + epsilon)

        Esto centra cada feature en 0 con varianza 1.

        Complejidad: O(N * F)

        Parametros:
        - X: datos de entrada (N x F)
        - epsilon: termino de estabilidad numerica

        Retorna:
        - X_norm: datos normalizados (N x F)
        """
        n = X.rows
        f = X.cols
        result = Matrix(n, f)

        for j in range(f):
            # Media de la columna
            mean_j = sum(X.data[i][j] for i in range(n)) / n

            # Desviacion estandar (poblacional)
            var_j = sum((X.data[i][j] - mean_j) ** 2 for i in range(n)) / n
            std_j = math.sqrt(var_j)

            # Normalizar
            for i in range(n):
                result.data[i][j] = (X.data[i][j] - mean_j) / (std_j + epsilon)

        return result

    # ----------------------------------------------------------
    # Division Train / Validation
    # ----------------------------------------------------------

    @staticmethod
    def split_dataset(X: Matrix, y: Matrix,
                      train_ratio: float = 0.8,
                      seed: Optional[int] = None) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """
        Divide el dataset en entrenamiento y validacion.

        Mezcla aleatoriamente los indices antes de dividir.

        Parametros:
        - X: features (N x F)
        - y: etiquetas (N x C)
        - train_ratio: fraccion para entrenamiento (0 < ratio < 1)
        - seed: semilla para reproducibilidad

        Retorna:
        - X_train, y_train, X_val, y_val

        Complejidad: O(N)
        """
        if seed is not None:
            random.seed(seed)

        n = X.rows
        indices = list(range(n))
        random.shuffle(indices)

        split_point = int(n * train_ratio)
        train_idx = indices[:split_point]
        val_idx = indices[split_point:]

        n_train = len(train_idx)
        n_val = len(val_idx)

        X_train = Matrix(n_train, X.cols, [X.data[i][:] for i in train_idx])
        y_train = Matrix(n_train, y.cols, [y.data[i][:] for i in train_idx])

        X_val = Matrix(n_val, X.cols, [X.data[i][:] for i in val_idx])
        y_val = Matrix(n_val, y.cols, [y.data[i][:] for i in val_idx])

        return X_train, y_train, X_val, y_val

    # ----------------------------------------------------------
    # Utilidades adicionales
    # ----------------------------------------------------------

    @staticmethod
    def one_hot_to_labels(y: Matrix) -> List[int]:
        """
        Convierte matriz one-hot a lista de etiquetas enteras.
        Complejidad: O(N * C)
        """
        labels = []
        for i in range(y.rows):
            clase = max(range(y.cols), key=lambda j: y.data[i][j])
            labels.append(clase)
        return labels

    @staticmethod
    def labels_to_one_hot(labels: List[int], num_classes: int) -> Matrix:
        """
        Convierte lista de etiquetas a matriz one-hot.
        Complejidad: O(N * C)
        """
        n = len(labels)
        y = Matrix(n, num_classes)
        for i, c in enumerate(labels):
            y.data[i][c] = 1.0
        return y
