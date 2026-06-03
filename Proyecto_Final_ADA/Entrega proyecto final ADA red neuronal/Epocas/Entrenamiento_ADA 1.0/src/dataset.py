"""
Generador de datasets para entrenar la red neuronal
Incluye MNIST simplificado y datos sintéticos
"""

import math
import random
from typing import Tuple, List
from src.matrix import Matrix


class DatasetGenerator:
    """
    Generador de datasets para clasificación
    Complejidad: Depende del tamaño del dataset
    """
    
    @staticmethod
    def generate_mnist_like(num_samples: int = 1000, num_classes: int = 10,
                           img_size: int = 28, seed: int = 42) -> Tuple[Matrix, Matrix]:
        """
        Genera un dataset similar a MNIST - DESAFIANTE y REALISTA
        Crea imágenes sintéticas de dígitos con patrones complejos y ruido
        
        Complejidad temporal: O(num_samples * img_size²)
        Complejidad espacial: O(num_samples * (img_size² + num_classes))
        
        Retorna: (X, y) donde X es (num_samples, img_size²) e y es (num_samples, num_classes)
        """
        if seed is not None:
            random.seed(seed)
        
        img_features = img_size * img_size
        
        # Crear datos
        X_data = [[0.0 for _ in range(img_features)] for _ in range(num_samples)]
        y_data = [[0.0 for _ in range(num_classes)] for _ in range(num_samples)]
        
        for sample_idx in range(num_samples):
            # Asignar clase aleatoria
            true_class = sample_idx % num_classes
            
            # Ruido de fondo alto (dataset desafiante)
            for pixel_idx in range(img_features):
                X_data[sample_idx][pixel_idx] = random.uniform(0, 0.4)
            
            # Crear patrón específico de la clase (estructura característica)
            # Cada clase tiene un patrón único más complejo
            row_center = img_size // 2
            col_center = img_size // 2
            
            # Patrón 1: Centro específico por clase
            radius_base = img_size // 4
            for pixel_idx in range(img_features):
                row = pixel_idx // img_size
                col = pixel_idx % img_size
                
                # Distancia al centro ponderada por clase
                dist = math.sqrt((row - row_center)**2 + (col - col_center)**2)
                
                # Patrón único por clase usando diferentes radios y pesos
                class_factor = (true_class + 1) / num_classes
                pattern_value = math.exp(-dist / (radius_base * class_factor))
                
                # Agregar patrón al píxel (pero no simple, usar múltiples capas)
                current = X_data[sample_idx][pixel_idx]
                X_data[sample_idx][pixel_idx] = min(1.0, current + pattern_value * 0.6)
            
            # Patrón 2: Línea diagonal única por clase (para mayor variabilidad)
            for pixel_idx in range(img_features):
                row = pixel_idx // img_size
                col = pixel_idx % img_size
                
                # Diagonal offset por clase
                offset = (true_class - num_classes // 2) * 2
                if abs(row - col - offset) < 2:
                    X_data[sample_idx][pixel_idx] = min(1.0, 
                        X_data[sample_idx][pixel_idx] + 0.3)
            
            # Patrón 3: Esquina específica por clase
            if true_class < 4:  # Esquinas arriba
                corner_row = img_size // 4
                corner_col = (true_class % 2) * 3 * img_size // 4
            else:  # Esquinas abajo
                corner_row = 3 * img_size // 4
                corner_col = ((true_class - 4) % 2) * 3 * img_size // 4
            
            for pixel_idx in range(img_features):
                row = pixel_idx // img_size
                col = pixel_idx % img_size
                dist_corner = math.sqrt((row - corner_row)**2 + 
                                       (col - corner_col)**2)
                if dist_corner < img_size // 3:
                    X_data[sample_idx][pixel_idx] = max(0.0, 
                        X_data[sample_idx][pixel_idx] - 0.2)
            
            # Agregar ruido gaussiano variable (hace dataset más difícil)
            for pixel_idx in range(img_features):
                noise = random.gauss(0, 0.1)
                X_data[sample_idx][pixel_idx] = max(0, min(1, 
                    X_data[sample_idx][pixel_idx] + noise))
            
            # One-hot encoding para y
            y_data[sample_idx][true_class] = 1.0
        
        return Matrix(num_samples, img_features, X_data), \
               Matrix(num_samples, num_classes, y_data)
    
    @staticmethod
    def generate_synthetic_classification(num_samples: int = 1000, 
                                         num_features: int = 20,
                                         num_classes: int = 3,
                                         seed: int = 42) -> Tuple[Matrix, Matrix]:
        """
        Genera datos de clasificación sintéticos
        Crea clusters gaussianos por clase
        
        Complejidad temporal: O(num_samples * num_features)
        Complejidad espacial: O(num_samples * (num_features + num_classes))
        """
        if seed is not None:
            random.seed(seed)
        
        X_data = [[0.0 for _ in range(num_features)] for _ in range(num_samples)]
        y_data = [[0.0 for _ in range(num_classes)] for _ in range(num_samples)]
        
        samples_per_class = num_samples // num_classes
        
        for class_idx in range(num_classes):
            # Centro del cluster para esta clase
            center = [random.uniform(-5, 5) for _ in range(num_features)]
            
            # Generar muestras de esta clase
            start_idx = class_idx * samples_per_class
            end_idx = start_idx + samples_per_class
            
            if class_idx == num_classes - 1:
                end_idx = num_samples  # Última clase toma el resto
            
            for sample_idx in range(start_idx, end_idx):
                # Generar punto alrededor del centro (distribución gaussiana)
                for feat_idx in range(num_features):
                    X_data[sample_idx][feat_idx] = center[feat_idx] + random.gauss(0, 1.0)
                
                # One-hot encoding
                y_data[sample_idx][class_idx] = 1.0
        
        return Matrix(num_samples, num_features, X_data), \
               Matrix(num_samples, num_classes, y_data)
    
    @staticmethod
    def generate_xor_dataset(num_samples: int = 100, seed: int = 42) -> Tuple[Matrix, Matrix]:
        """
        Genera dataset XOR (problema no-linealmente separable)
        Útil para probar que la red puede aprender funciones no-lineales
        
        Complejidad temporal: O(num_samples)
        Complejidad espacial: O(num_samples * 3)
        """
        if seed is not None:
            random.seed(seed)
        
        X_data = [[0.0, 0.0] for _ in range(num_samples)]
        y_data = [[0.0, 0.0] for _ in range(num_samples)]
        
        for sample_idx in range(num_samples):
            # Generar puntos aleatorios
            x1 = random.uniform(-1, 1)
            x2 = random.uniform(-1, 1)
            
            X_data[sample_idx][0] = x1
            X_data[sample_idx][1] = x2
            
            # XOR lógico
            if (x1 > 0 and x2 > 0) or (x1 < 0 and x2 < 0):
                y_data[sample_idx][1] = 1.0  # Clase 1
            else:
                y_data[sample_idx][0] = 1.0  # Clase 0
        
        return Matrix(num_samples, 2, X_data), \
               Matrix(num_samples, 2, y_data)
    
    @staticmethod
    def split_dataset(X: Matrix, y: Matrix, train_ratio: float = 0.8, 
                     seed: int = 42) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """
        Divide el dataset en entrenamiento y validación
        
        Complejidad temporal: O(num_samples) para crear índices + copias
        Complejidad espacial: O(num_samples)
        """
        if seed is not None:
            random.seed(seed)
        
        num_samples = X.rows
        train_size = int(num_samples * train_ratio)
        
        # Crear índices permutados
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Crear matrices para entrenamiento
        X_train = Matrix.zeros(len(train_indices), X.cols)
        y_train = Matrix.zeros(len(train_indices), y.cols)
        
        for i, idx in enumerate(train_indices):
            X_train.data[i] = X.data[idx][:]
            y_train.data[i] = y.data[idx][:]
        
        # Crear matrices para validación
        X_val = Matrix.zeros(len(val_indices), X.cols)
        y_val = Matrix.zeros(len(val_indices), y.cols)
        
        for i, idx in enumerate(val_indices):
            X_val.data[i] = X.data[idx][:]
            y_val.data[i] = y.data[idx][:]
        
        return X_train, y_train, X_val, y_val
    
    @staticmethod
    def normalize_dataset(X: Matrix) -> Matrix:
        """
        Normaliza el dataset a media 0 y desviación estándar 1
        Normalización Z-score
        
        Complejidad temporal: O(num_samples * num_features * 2)
        = O(num_samples * num_features)
        Complejidad espacial: O(num_features) para medias y desviaciones
        """
        num_samples = X.rows
        num_features = X.cols
        
        # Calcular media
        means = [0.0] * num_features
        for j in range(num_features):
            for i in range(num_samples):
                means[j] += X.data[i][j]
            means[j] /= num_samples
        
        # Calcular desviación estándar
        stds = [0.0] * num_features
        for j in range(num_features):
            sum_sq_diff = 0.0
            for i in range(num_samples):
                diff = X.data[i][j] - means[j]
                sum_sq_diff += diff * diff
            stds[j] = math.sqrt(sum_sq_diff / num_samples)
        
        # Normalizar
        X_norm = X.copy()
        for i in range(num_samples):
            for j in range(num_features):
                if stds[j] > 0:
                    X_norm.data[i][j] = (X.data[i][j] - means[j]) / stds[j]
                else:
                    X_norm.data[i][j] = 0.0
        
        return X_norm
