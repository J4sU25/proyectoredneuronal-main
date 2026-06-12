"""
Red Neuronal Multicapa (MLP) implementada desde cero.
Sin uso de NumPy ni frameworks de alto nivel.

Arquitectura:
- Capas densas (fully connected) apiladas
- Funciones de activacion: ReLU, Sigmoid, Tanh, Lineal
- Softmax en la ultima capa para clasificacion
- Funcion de perdida: Cross-Entropy

Backpropagation:
- Diferenciacion automatica manual (regla de la cadena)
- Actualizacion SGD (Stochastic Gradient Descent) con mini-batches

Complejidad temporal:
- Forward (por lote B): O(B * sum(I_i * O_i))
- Backward (por lote B): O(B * sum(I_i * O_i))  ~ 1.5x forward
- Total por epoca: O(E/B * B * sum(I_i * O_i)) = O(E * sum(I_i * O_i))

Donde I_i = entradas capa i, O_i = salidas capa i
"""

import math
import random
import json
import os
from typing import List, Tuple, Optional

from src.matrix import Matrix
from src.activation import ActivationFunction, softmax


# ============================================================
# CAPA DENSA (Dense Layer)
# ============================================================

class DenseLayer:
    """
    Capa densa (fully connected).

    Parametros aprendibles:
    - W: matriz de pesos (input_size x output_size), inicializada con He
    - b: vector de sesgos (1 x output_size), inicializado en ceros

    Cache para backpropagation:
    - input_cache:  entrada a la capa (X)
    - output_cache: pre-activacion (Z = X @ W + b)
    - activ_cache:  activacion aplicada (A = activation(Z))

    Complejidad espacial: O(input_size * output_size)
    """

    def __init__(self, input_size: int, output_size: int, activation: str):
        """
        Inicializa la capa con pesos He (recomendado para ReLU).

        Inicializacion He: std = sqrt(2 / input_size)
        Para 'sigmoid'/'tanh' se usa Xavier: std = sqrt(1 / input_size)

        Complejidad: O(input_size * output_size)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation.lower()

        # Inicializacion de pesos
        if self.activation in ('relu',):
            std = math.sqrt(2.0 / input_size)  # He
        else:
            std = math.sqrt(1.0 / input_size)  # Xavier

        self.W = Matrix.random_normal(input_size, output_size, mean=0.0, std=std)
        self.b = Matrix.zeros(1, output_size)

        # Cache para backpropagation
        self.input_cache: Optional[Matrix] = None   # X de entrada
        self.output_cache: Optional[Matrix] = None  # Z = XW + b (pre-activacion)
        self.activ_cache: Optional[Matrix] = None   # A = activation(Z)

    def forward(self, X: Matrix) -> Matrix:
        """
        Paso hacia adelante.

        Z = X @ W + b  (broadcasting del sesgo)
        A = activation(Z)

        Complejidad: O(batch_size * input_size * output_size)

        Parametros:
        - X: entrada (batch_size x input_size)

        Retorna:
        - A: activaciones (batch_size x output_size)
        """
        # Guardar entrada para backprop
        self.input_cache = X

        # Z = X @ W + b
        Z = X.dot(self.W).add(self.b)

        # Guardar pre-activacion para backprop
        self.output_cache = Z

        # Aplicar activacion
        A = ActivationFunction.apply(Z, self.activation)
        self.activ_cache = A

        return A

    def backward(self, dA: Matrix, learning_rate: float) -> Matrix:
        """
        Paso hacia atras (backpropagation).

        Calcula gradientes y actualiza pesos con SGD.

        Gradientes:
        - dZ = dA * activation'(Z)           [elemento a elemento]
        - dW = (X^T @ dZ) / batch_size
        - db = sum(dZ, axis=0) / batch_size
        - dX = dZ @ W^T                      [para propagar a la capa anterior]

        Actualizacion SGD:
        - W = W - lr * dW
        - b = b - lr * db

        Complejidad: O(batch_size * input_size * output_size)

        Parametros:
        - dA: gradiente entrante desde la capa superior (batch_size x output_size)
        - learning_rate: tasa de aprendizaje

        Retorna:
        - dX: gradiente para la capa anterior (batch_size x input_size)
        """
        batch_size = dA.rows

        # dZ = dA * activation'(Z)
        dZ_activation = ActivationFunction.apply_derivative(
            self.output_cache, self.activation
        )
        dZ = dA.element_wise_mult(dZ_activation)

        # dW = X^T @ dZ / batch_size
        X_T = self.input_cache.transpose()
        dW = X_T.dot(dZ).scalar_mult(1.0 / batch_size)

        # db = mean(dZ, axis=0)
        db = Matrix.zeros(1, self.output_size)
        for j in range(self.output_size):
            s = 0.0
            for i in range(batch_size):
                s += dZ.data[i][j]
            db.data[0][j] = s / batch_size

        # dX = dZ @ W^T (para propagar a la capa anterior)
        W_T = self.W.transpose()
        dX = dZ.dot(W_T)

        # Actualizar pesos: SGD
        self.W = self.W.subtract(dW.scalar_mult(learning_rate))
        self.b = self.b.subtract(db.scalar_mult(learning_rate))

        return dX

    def get_params_count(self) -> int:
        """Retorna el numero total de parametros de la capa."""
        return self.input_size * self.output_size + self.output_size


# ============================================================
# RED NEURONAL MULTICAPA (MLP)
# ============================================================

class MLP:
    """
    Red Neuronal Multicapa (Multi-Layer Perceptron).

    Implementacion desde cero sin frameworks externos.

    Arquitectura:
    - Capas ocultas con activaciones configurables (relu, sigmoid, tanh, linear)
    - Ultima capa: lineal + Softmax para clasificacion multiclase
    - Perdida: Cross-Entropy

    Uso tipico:
        model = MLP([784, 64, 64, 10], ['relu', 'relu', 'linear'], 'cross_entropy')
        for epoch in range(100):
            loss, acc = model.train_epoch(X, y, batch_size=32, learning_rate=0.01)
        preds = model.forward(X_test)

    Complejidad por epoca:
        O(N/B * B * sum(I_i * O_i)) = O(N * sum(I_i * O_i))
    """

    def __init__(self,
                 layer_sizes: List[int],
                 activations: List[str],
                 loss_function: str = 'cross_entropy'):
        """
        Inicializa el MLP.

        Parametros:
        - layer_sizes: lista con el tamano de cada capa.
                       Ejemplo: [784, 64, 10] -> capa entrada 784, oculta 64, salida 10
        - activations: lista de activaciones para cada capa oculta/salida.
                       Debe tener len(layer_sizes) - 1 elementos.
                       Ejemplo: ['relu', 'linear']
        - loss_function: funcion de perdida ('cross_entropy')

        Lanza ValueError si las dimensiones son inconsistentes.
        """
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Numero de activaciones ({len(activations)}) debe ser "
                f"igual a numero de capas - 1 ({len(layer_sizes) - 1})"
            )

        self.layer_sizes = layer_sizes
        self.loss_function = loss_function.lower()

        # Crear capas densas
        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    def forward(self, X: Matrix) -> Matrix:
        """
        Propaga la entrada hacia adelante.

        Las capas ocultas aplican su activacion.
        La ultima capa aplica Softmax para clasificacion.

        Complejidad: O(batch_size * sum(I_i * O_i))

        Parametros:
        - X: entrada (batch_size x input_size)

        Retorna:
        - probabilidades: (batch_size x num_classes), suma por fila = 1
        """
        A = X
        for i, layer in enumerate(self.layers):
            A = layer.forward(A)

        # Softmax en la salida para clasificacion multiclase
        return softmax(A)

    # ----------------------------------------------------------
    # Perdida (Loss)
    # ----------------------------------------------------------

    def compute_loss(self, predictions: Matrix, targets: Matrix) -> float:
        """
        Calcula la cross-entropy promediada sobre el lote.

        L = -1/N * sum_i sum_c [y_ic * log(p_ic)]

        Donde:
        - y_ic = etiqueta one-hot verdadera
        - p_ic = probabilidad predicha

        Estabilidad numerica: clip predicciones en [1e-10, 1]

        Complejidad: O(batch_size * num_classes)
        """
        if self.loss_function != 'cross_entropy':
            raise ValueError(f"Funcion de perdida desconocida: '{self.loss_function}'")

        n = predictions.rows
        total_loss = 0.0

        for i in range(n):
            for c in range(predictions.cols):
                # Clip para estabilidad numerica
                p = max(predictions.data[i][c], 1e-10)
                total_loss -= targets.data[i][c] * math.log(p)

        return total_loss / n

    # ----------------------------------------------------------
    # Precision (Accuracy)
    # ----------------------------------------------------------

    def compute_accuracy(self, predictions: Matrix, targets: Matrix) -> float:
        """
        Calcula la precision: fraccion de predicciones correctas.

        Argmax de predicciones vs argmax de targets.

        Complejidad: O(batch_size * num_classes)
        """
        correct = 0
        for i in range(predictions.rows):
            pred_class = max(range(predictions.cols),
                             key=lambda j: predictions.data[i][j])
            true_class = max(range(targets.cols),
                             key=lambda j: targets.data[i][j])
            if pred_class == true_class:
                correct += 1
        return correct / predictions.rows

    # ----------------------------------------------------------
    # Backward Pass
    # ----------------------------------------------------------

    def backward(self, dLoss: Matrix, learning_rate: float) -> None:
        """
        Propaga el gradiente hacia atras y actualiza los pesos.

        Para la ultima capa, el gradiente ya fue calculado como
        dL/dZ = softmax_derivative(predictions, targets).

        El backward de cada capa calcula dX para propagar
        a la capa anterior.

        Complejidad: O(batch_size * sum(I_i * O_i))

        Parametros:
        - dLoss: gradiente de la perdida respecto a la salida de la ultima capa
                 (batch_size x num_classes)
        - learning_rate: tasa de aprendizaje para SGD
        """
        dA = dLoss
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    # ----------------------------------------------------------
    # Entrenamiento por Epoca
    # ----------------------------------------------------------

    def train_epoch(self, X: Matrix, y: Matrix,
                    batch_size: int = 32,
                    learning_rate: float = 0.01) -> Tuple[float, float]:
        """
        Entrena el modelo por una epoca completa usando mini-batches.

        Algoritmo por cada mini-batch:
        1. Forward pass -> probabilidades
        2. Calcular perdida
        3. Calcular gradiente del softmax+CE
        4. Backward pass -> actualizar pesos

        Los indices se barajan aleatoriamente cada epoca.

        Complejidad: O(N/B * B * sum(I_i * O_i)) = O(N * sum(I_i * O_i))

        Parametros:
        - X: datos de entrenamiento (N x input_size)
        - y: etiquetas one-hot (N x num_classes)
        - batch_size: tamano del mini-lote (B)
        - learning_rate: tasa de aprendizaje

        Retorna:
        - (loss_promedio, accuracy_promedio) de la epoca
        """
        from src.activation import softmax_derivative

        n = X.rows
        total_loss = 0.0
        total_correct = 0
        num_batches = 0

        # Barajar indices (SGD)
        indices = list(range(n))
        random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            actual_batch = len(batch_idx)

            # Extraer mini-batch
            X_batch = Matrix(actual_batch, X.cols,
                             [X.data[i][:] for i in batch_idx])
            y_batch = Matrix(actual_batch, y.cols,
                             [y.data[i][:] for i in batch_idx])

            # Forward pass
            preds = self.forward(X_batch)

            # Perdida del batch
            batch_loss = self.compute_loss(preds, y_batch)
            total_loss += batch_loss

            # Precision del batch
            total_correct += int(self.compute_accuracy(preds, y_batch) * actual_batch)

            # Gradiente combinado softmax + cross-entropy
            dLoss = softmax_derivative(preds, y_batch)

            # Backward pass
            self.backward(dLoss, learning_rate)

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_correct / n if n > 0 else 0.0

        return avg_loss, avg_acc

    # ----------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------

    def get_total_params(self) -> int:
        """Retorna el numero total de parametros del modelo."""
        return sum(layer.get_params_count() for layer in self.layers)

    # ----------------------------------------------------------
    # Persistencia (Guardar / Cargar Pesos)
    # ----------------------------------------------------------

    def save_weights(self, filepath: str) -> bool:
        """
        Guarda los pesos del modelo en formato JSON.

        Formato:
        {
            "layer_sizes": [...],
            "layers": [
                {"W": [[...]], "b": [[...]], "activation": "relu"},
                ...
            ]
        }

        Complejidad: O(total_params)

        Retorna: True si exitoso, False si hay error.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True) \
                if os.path.dirname(filepath) else None

            data = {
                "layer_sizes": self.layer_sizes,
                "loss_function": self.loss_function,
                "layers": []
            }

            for layer in self.layers:
                layer_data = {
                    "activation": layer.activation,
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "W": layer.W.data,
                    "b": layer.b.data
                }
                data["layers"].append(layer_data)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)

            return True

        except Exception as e:
            print(f"Error guardando pesos: {e}")
            return False

    def load_weights(self, filepath: str) -> bool:
        """
        Carga los pesos del modelo desde un archivo JSON.

        Compatible con multiples formatos:
        - Formato actual:   {"layer_sizes": [...], "layers": [{"W":..., "b":...}]}
        - Formato legado:   {"weights": [...], "biases": [...]}
        - Formato legado 2: {"capas": [...]}

        Complejidad: O(total_params)

        Retorna: True si exitoso, False si hay error.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # -- Detectar formato del archivo --

            # Formato actual: clave "layers"
            if "layers" in data:
                layers_data = data["layers"]
                if len(layers_data) != len(self.layers):
                    print(f"Advertencia: numero de capas diferente en checkpoint "
                          f"({len(layers_data)}) vs modelo ({len(self.layers)}). "
                          f"Ignorando pesos y empezando desde cero.")
                    return False

                for i, layer_data in enumerate(layers_data):
                    layer = self.layers[i]
                    w_data = layer_data["W"]
                    b_data = layer_data["b"]
                    layer.W = Matrix(len(w_data), len(w_data[0]), w_data)
                    layer.b = Matrix(len(b_data), len(b_data[0]), b_data)
                return True

            # Formato legado: claves "weights" y "biases" como listas planas
            elif "weights" in data and "biases" in data:
                weights_list = data["weights"]
                biases_list  = data["biases"]
                if len(weights_list) != len(self.layers):
                    print(f"Advertencia: checkpoint legado con {len(weights_list)} capas "
                          f"vs modelo con {len(self.layers)} capas. Empezando desde cero.")
                    return False
                for i, layer in enumerate(self.layers):
                    w_data = weights_list[i]
                    b_data = biases_list[i]
                    layer.W = Matrix(len(w_data), len(w_data[0]), w_data)
                    if isinstance(b_data[0], list):
                        layer.b = Matrix(len(b_data), len(b_data[0]), b_data)
                    else:
                        layer.b = Matrix(1, len(b_data), [b_data])
                return True

            # Formato desconocido: listar claves para diagnostico
            else:
                claves = list(data.keys())
                print(f"Error cargando pesos: el archivo no tiene el formato esperado.\n"
                      f"  Claves encontradas: {claves}\n"
                      f"  Se esperaba al menos una de: 'layers', 'weights'\n"
                      f"  El modelo arrancara con pesos aleatorios.")
                return False

        except json.JSONDecodeError as e:
            print(f"Error cargando pesos: el archivo no es JSON valido -> {e}")
            return False
        except Exception as e:
            print(f"Error cargando pesos: {e}")
            return False
