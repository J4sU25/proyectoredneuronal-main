"""
Clasificador k-NN (k-Nearest Neighbors) - Implementacion desde cero
Sin uso de frameworks de alto nivel (solo Python estandar)

Fase 4: Baseline k-NN para comparacion con MLP
"""

import math
import time
from typing import List, Tuple, Optional
from src.matrix import Matrix


# ============================================================
# DISTANCIA EUCLIDIANA
# Complejidad: O(d) donde d = numero de dimensiones
# ============================================================

def distancia_euclidiana(v1: List[float], v2: List[float]) -> float:
    """
    Calcula la distancia Euclidiana entre dos vectores.

    dist(v1, v2) = sqrt( sum_i (v1[i] - v2[i])^2 )

    Complejidad temporal: O(d) donde d = len(v1) = len(v2)
    Complejidad espacial: O(1)
    """
    suma = 0.0
    for i in range(len(v1)):
        diff = v1[i] - v2[i]
        suma += diff * diff
    return math.sqrt(suma)


# ============================================================
# HEAP MINIMO PARA TOP-K VECINOS
# Permite mantener los k vecinos mas cercanos en O(log k)
# ============================================================

class _MinHeapKNN:
    """
    Min-Heap auxiliar para k-NN.
    Almacena pares (distancia, clase) para los k vecinos mas cercanos.

    Estrategia: Max-Heap de tamano k
    - Si el nuevo vecino es mas cercano que el mas lejano en el heap,
      se reemplaza -> mantiene los k mas cercanos en O(log k)

    Complejidad por insercion: O(log k)
    Complejidad total para N vecinos: O(N log k)
    vs. ordenar todos y tomar k: O(N log N)
    """

    def __init__(self):
        # (distancia_negativa, clase) - negativo para simular max-heap con min-heap
        self.heap: List[Tuple[float, int]] = []

    def _heapify_up(self, idx: int) -> None:
        """Sube el elemento a su posicion correcta. O(log k)"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] > self.heap[parent][0]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break

    def _heapify_down(self, idx: int) -> None:
        """Baja el elemento a su posicion correcta. O(log k)"""
        n = len(self.heap)
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left < n and self.heap[left][0] > self.heap[largest][0]:
                largest = left
            if right < n and self.heap[right][0] > self.heap[largest][0]:
                largest = right

            if largest != idx:
                self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
                idx = largest
            else:
                break

    def push(self, dist: float, clase: int, k: int) -> None:
        """
        Agrega un vecino manteniendo solo los k mas cercanos.
        Si el heap tiene menos de k elementos, agrega directamente.
        Si ya tiene k, reemplaza si la nueva distancia es menor a la mayor.

        Complejidad: O(log k)
        """
        if len(self.heap) < k:
            self.heap.append((dist, clase))
            self._heapify_up(len(self.heap) - 1)
        elif dist < self.heap[0][0]:
            # Reemplazar el mas lejano (raiz del max-heap)
            self.heap[0] = (dist, clase)
            self._heapify_down(0)

    def get_all(self) -> List[Tuple[float, int]]:
        """Retorna todos los elementos del heap."""
        return self.heap[:]


# ============================================================
# CLASIFICADOR k-NN
# ============================================================

class KNNClassifier:
    """
    Clasificador k-Nearest Neighbors implementado desde cero.

    Complejidad temporal:
    - fit: O(1) - solo almacena los datos
    - predict (para 1 muestra): O(N * d + N log k)
        O(N * d) para calcular N distancias
        O(N log k) para mantener los k vecinos con heap
    - predict (para M muestras): O(M * N * d)

    Complejidad espacial: O(N * d) para almacenar el dataset

    Donde:
    - N = numero de muestras de entrenamiento
    - d = dimensionalidad de los datos
    - k = numero de vecinos
    - M = numero de muestras a clasificar
    """

    def __init__(self, k: int = 5):
        """
        Inicializa el clasificador.

        Parametros:
        - k: numero de vecinos mas cercanos a considerar
        """
        self.k = k
        self.X_train: Optional[List[List[float]]] = None
        self.y_train: Optional[List[int]] = None
        self.num_classes: int = 0

    def fit(self, X: Matrix, y: Matrix) -> None:
        """
        Almacena el dataset de entrenamiento.

        Complejidad temporal: O(N) para convertir etiquetas one-hot a indices
        Complejidad espacial: O(N * d)
        """
        self.X_train = X.data  # Lista de listas

        # Convertir one-hot a indices de clase
        self.y_train = []
        for i in range(y.rows):
            # Argmax de one-hot encoding
            clase = max(range(y.cols), key=lambda j: y.data[i][j])
            self.y_train.append(clase)

        self.num_classes = y.cols

    def _predecir_muestra(self, x: List[float]) -> int:
        """
        Predice la clase de una sola muestra usando k-NN con heap.

        Algoritmo:
        1. Calcular distancias a todos los puntos de entrenamiento: O(N * d)
        2. Mantener k vecinos mas cercanos con max-heap: O(N log k)
        3. Votacion por mayoria entre los k vecinos: O(k)

        Complejidad total: O(N * d + N log k) ~ O(N * d)
        """
        heap = _MinHeapKNN()

        # Calcular distancias y mantener k vecinos mas cercanos
        for i, x_train in enumerate(self.X_train):
            dist = distancia_euclidiana(x, x_train)
            heap.push(dist, self.y_train[i], self.k)

        # Votacion por mayoria
        votos = [0] * self.num_classes
        for dist, clase in heap.get_all():
            votos[clase] += 1

        # Retornar clase con mas votos
        return max(range(self.num_classes), key=lambda j: votos[j])

    def predict(self, X: Matrix) -> List[int]:
        """
        Predice las clases para un conjunto de muestras.

        Complejidad temporal: O(M * N * d)
        Complejidad espacial: O(M) para los resultados
        """
        if self.X_train is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a fit() primero.")

        predicciones = []
        for i in range(X.rows):
            pred = self._predecir_muestra(X.data[i])
            predicciones.append(pred)

        return predicciones

    def compute_accuracy(self, X: Matrix, y: Matrix) -> float:
        """
        Calcula la precision del clasificador.

        Complejidad temporal: O(M * N * d)
        Complejidad espacial: O(1)
        """
        predicciones = self.predict(X)
        correctos = 0

        for i in range(y.rows):
            clase_real = max(range(y.cols), key=lambda j: y.data[i][j])
            if predicciones[i] == clase_real:
                correctos += 1

        return correctos / y.rows


# ============================================================
# COMPARATIVA k-NN vs MLP
# ============================================================

def comparar_knn_vs_mlp(X_train: Matrix, y_train: Matrix,
                         X_val: Matrix, y_val: Matrix,
                         mlp_model,
                         k_values: List[int] = None) -> dict:
    """
    Compara el baseline k-NN contra el MLP entrenado.

    Para cada valor de k:
    - Entrena k-NN (O(1) - solo guarda datos)
    - Predice en validacion (O(M * N * d))
    - Compara con MLP

    Complejidad total: O(len(k_values) * M * N * d + MLP_inference)

    Retorna: diccionario con resultados de comparacion
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 11]

    results = {
        'k_values': k_values,
        'knn_accuracies': [],
        'knn_times': [],
        'mlp_accuracy': None,
        'mlp_time': None
    }

    # --- Evaluar MLP ---
    inicio = time.perf_counter()
    mlp_preds = mlp_model.forward(X_val)
    mlp_acc = mlp_model.compute_accuracy(mlp_preds, y_val)
    fin = time.perf_counter()

    results['mlp_accuracy'] = mlp_acc
    results['mlp_time'] = fin - inicio

    # --- Evaluar k-NN para cada k ---
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)

        inicio = time.perf_counter()
        acc = knn.compute_accuracy(X_val, y_val)
        fin = time.perf_counter()

        results['knn_accuracies'].append(acc)
        results['knn_times'].append(fin - inicio)

    return results


def imprimir_comparativa_knn(results: dict) -> None:
    """Imprime tabla de comparacion k-NN vs MLP."""
    print("\n" + "=" * 80)
    print("  FASE 4: COMPARATIVA k-NN vs MLP")
    print("=" * 80)

    print("\nAnalisis de Complejidad:")
    print("  k-NN:")
    print("    - Entrenamiento: O(1) - no aprende, solo memoriza")
    print("    - Prediccion 1 muestra: O(N*d + N*log(k)) = O(N*d)")
    print("    - Prediccion M muestras: O(M*N*d)")
    print("    - Espacio: O(N*d) para almacenar dataset")
    print("  MLP:")
    print("    - Entrenamiento: O(E * N/B * sum(I_i * O_i))")
    print("    - Prediccion M muestras: O(M * sum(I_i * O_i))")
    print("    - Espacio: O(sum(I_i * O_i)) para pesos")

    print("\n  Ventaja k-NN: No requiere entrenamiento")
    print("  Desventaja k-NN: Prediccion lenta (O(N*d) vs O(sum(I_i*O_i)) del MLP)")
    print("  Ventaja MLP: Prediccion rapida; aprende representaciones")

    print("\n" + "-" * 80)
    print(f"  {'Modelo':>20} | {'k':>4} | {'Precision':>10} | {'Tiempo pred.':>14}")
    print("-" * 80)

    mlp_acc = results['mlp_accuracy']
    mlp_time = results['mlp_time']
    print(f"  {'MLP (red neuronal)':>20} | {'--':>4} | {mlp_acc:.4f}     | {mlp_time*1000:>10.4f} ms")

    for i, k in enumerate(results['k_values']):
        acc = results['knn_accuracies'][i]
        t = results['knn_times'][i]
        print(f"  {f'k-NN (k={k})':>20} | {k:>4} | {acc:.4f}     | {t*1000:>10.4f} ms")

    print("-" * 80)

    # Conclusion
    best_knn_acc = max(results['knn_accuracies'])
    best_k = results['k_values'][results['knn_accuracies'].index(best_knn_acc)]

    print(f"\n  Mejor k-NN: k={best_k}, Precision={best_knn_acc:.4f}")
    print(f"  MLP: Precision={mlp_acc:.4f}")

    if mlp_acc > best_knn_acc:
        mejora = (mlp_acc - best_knn_acc) * 100
        print(f"\n  -> MLP supera al mejor k-NN en {mejora:.2f} puntos porcentuales")
        print(f"  -> El aprendizaje de representaciones del MLP es superior al k-NN")
    else:
        diff = (best_knn_acc - mlp_acc) * 100
        print(f"\n  -> k-NN supera al MLP en {diff:.2f} puntos porcentuales")
        print(f"  -> En este dataset, la similitud geometrica es un buen predictor")
