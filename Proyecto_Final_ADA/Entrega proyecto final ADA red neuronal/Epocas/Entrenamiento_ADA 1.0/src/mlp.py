"""
Red Neuronal Multicapa (MLP) - Implementación desde cero
Análisis completo de complejidad temporal y espacial
"""

import math
import random
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from src.matrix import Matrix
from src.activation import ActivationFunction, softmax, softmax_derivative
from src.data_structures import HashTable, MaxHeap, quickselect_top_k


class Layer:
    """
    Capa de una red neuronal
    
    Complejidad espacial: O(input_size * output_size + output_size)
    - W: O(input_size * output_size)
    - b: O(output_size)
    - gradW: O(input_size * output_size)
    - gradb: O(output_size)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Inicializa una capa
        Complejidad temporal: O(input_size * output_size)
        Complejidad espacial: O(input_size * output_size)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Inicialización He para ReLU
        self.W = Matrix.random(input_size, output_size, seed=None)
        self.b = Matrix.zeros(1, output_size)
        
        # Para propagación hacia atrás
        self.dW = None
        self.db = None
        self.input_cache = None
        self.output_cache = None
    
    def forward(self, X: Matrix) -> Matrix:
        """
        Propagación hacia adelante
        X: (batch_size, input_size)
        Salida: (batch_size, output_size)
        
        Complejidad temporal: O(batch_size * input_size * output_size)
        - Multiplicación matricial: O(B * I * O)
        - Suma de sesgo: O(B * O)
        - Activación: O(B * O)
        Total: O(B * I * O)
        
        Complejidad espacial: O(batch_size * output_size)
        """
        # Propagación lineal: X @ W + b
        Z = X.dot(self.W)
        
        # Sumar sesgo a cada fila
        for i in range(Z.rows):
            for j in range(Z.cols):
                Z.data[i][j] += self.b.data[0][j]
        
        # Aplicar activación
        A = ActivationFunction.apply_activation(Z, self.activation)
        
        # Cachear para backprop
        self.input_cache = X
        self.output_cache = Z  # Valores previos a activación
        
        return A
    
    def backward(self, dA: Matrix, learning_rate: float) -> Matrix:
        """
        Propagación hacia atrás
        dA: gradiente de la salida (batch_size, output_size)
        
        Complejidad temporal: O(batch_size * input_size * output_size)
        - Derivada activación: O(B * O)
        - Matriz transpuesta: O(O * B)
        - Multiplicación gradW: O(I * B * O)
        - Multiplicación dX: O(B * O * I)
        - Actualizar pesos: O(I * O + O)
        
        Complejidad espacial: O(batch_size * max(input_size, output_size))
        """
        batch_size = dA.rows
        
        # Derivada de activación
        dZ = ActivationFunction.apply_derivative(self.output_cache, self.activation)
        dZ = dZ.element_wise_mult(dA)
        
        # Gradiente de pesos: dW = X^T @ dZ / batch_size
        X_T = self.input_cache.transpose()
        self.dW = X_T.dot(dZ)
        self.dW = self.dW.scalar_mult(1.0 / batch_size)
        
        # Gradiente de sesgo: db = suma(dZ) / batch_size
        self.db = Matrix.zeros(1, self.output_size)
        for j in range(self.output_size):
            sum_grad = 0.0
            for i in range(batch_size):
                sum_grad += dZ.data[i][j]
            self.db.data[0][j] = sum_grad / batch_size
        
        # Gradiente para capa anterior: dX = dZ @ W^T
        W_T = self.W.transpose()
        dX = dZ.dot(W_T)
        
        # Actualizar pesos (SGD)
        self.W = self.W.subtract(self.dW.scalar_mult(learning_rate))
        self.b = self.b.subtract(self.db.scalar_mult(learning_rate))
        
        return dX


class MLP:
    """
    Red Neuronal Multicapa
    
    Complejidad espacial total: O(sum(input_i * output_i)) para todos los pesos
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[str], 
                 loss_function: str = 'cross_entropy'):
        """
        Inicializa la red
        layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
        activations: activación para cada capa oculta y de salida
        
        Complejidad temporal: O(sum(layer_sizes[i] * layer_sizes[i+1]))
        Complejidad espacial: O(sum(layer_sizes[i] * layer_sizes[i+1]))
        """
        self.layers: List[Layer] = []
        self.loss_function = loss_function
        
        # Crear capas
        for i in range(len(layer_sizes) - 1):
            activation = activations[i] if i < len(activations) else 'relu'
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activation)
            self.layers.append(layer)
        
        # Métricas
        self.training_losses: List[float] = []
        self.training_accuracies: List[float] = []
        self.epoch_times: List[float] = []
        
        # Estructuras para optimización
        self.loss_hash = HashTable()
    
    def forward(self, X: Matrix) -> Matrix:
        """
        Propagación hacia adelante completa
        X: (batch_size, input_features)
        Salida: (batch_size, num_classes)
        
        Complejidad temporal: O(sum(B * I_i * O_i)) para todas las capas
        donde I_i, O_i son dimensiones de capa i
        
        Complejidad espacial: O(B * max(I_i)) para almacenar activaciones
        """
        A = X
        for layer in self.layers[:-1]:
            A = layer.forward(A)
        
        # Última capa
        A = self.layers[-1].forward(A)
        
        # Aplicar softmax si es clasificación
        if self.loss_function == 'cross_entropy':
            A = softmax(A)
        
        return A
    
    def backward(self, dLoss: Matrix, learning_rate: float) -> None:
        """
        Propagación hacia atrás completa
        dLoss: gradiente de pérdida respecto a salida
        
        Complejidad temporal: O(sum(B * I_i * O_i))
        Complejidad espacial: O(B * max(I_i))
        """
        dA = dLoss
        
        # Propagar hacia atrás por todas las capas
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)
    
    def compute_loss(self, predictions: Matrix, targets: Matrix) -> float:
        """
        Calcula pérdida (cross-entropy)
        predictions: (batch_size, num_classes)
        targets: (batch_size, num_classes) one-hot encoded
        
        Complejidad temporal: O(batch_size * num_classes)
        Complejidad espacial: O(1)
        """
        batch_size = predictions.rows
        num_classes = predictions.cols
        total_loss = 0.0
        epsilon = 1e-7
        
        for i in range(batch_size):
            for j in range(num_classes):
                if targets.data[i][j] > 0:
                    pred = max(epsilon, min(1.0 - epsilon, predictions.data[i][j]))
                    total_loss -= targets.data[i][j] * math.log(pred)
        
        return total_loss / batch_size
    
    def compute_accuracy(self, predictions: Matrix, targets: Matrix) -> float:
        """
        Calcula precisión
        Complejidad temporal: O(batch_size * num_classes)
        Complejidad espacial: O(1)
        """
        batch_size = predictions.rows
        correct = 0
        
        for i in range(batch_size):
            # Argmax de predicción
            pred_class = max(range(predictions.cols), 
                           key=lambda j: predictions.data[i][j])
            # Argmax de target
            true_class = max(range(targets.cols),
                           key=lambda j: targets.data[i][j])
            
            if pred_class == true_class:
                correct += 1
        
        return correct / batch_size
    
    def train_epoch(self, X_train: Matrix, y_train: Matrix, 
                    batch_size: int, learning_rate: float) -> Tuple[float, float]:
        """
        Entrena una época
        X_train: (num_samples, input_features)
        y_train: (num_samples, num_classes)
        
        Complejidad temporal: O(num_samples/batch_size * (sum(B*I_i*O_i)))
        = O(num_samples * (sum(I_i*O_i) / (batch_size inherent)))
        
        En esencia: O(num_samples * param_count / batch_factor)
        
        Complejidad espacial: O(batch_size * max_layer_size + param_count)
        """
        num_samples = X_train.rows
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Índices aleatorios
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        # Procesar por lotes
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Extraer mini-batch
            X_batch = Matrix.zeros(len(batch_indices), X_train.cols)
            y_batch = Matrix.zeros(len(batch_indices), y_train.cols)
            
            for i, idx in enumerate(batch_indices):
                X_batch.data[i] = X_train.data[idx][:]
                y_batch.data[i] = y_train.data[idx][:]
            
            # Forward pass
            predictions = self.forward(X_batch)
            
            # Calcular pérdida y precisión
            batch_loss = self.compute_loss(predictions, y_batch)
            batch_accuracy = self.compute_accuracy(predictions, y_batch)
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            num_batches += 1
            
            # Backward pass
            if self.loss_function == 'cross_entropy':
                dLoss = softmax_derivative(predictions, y_batch)
            
            self.backward(dLoss, learning_rate)
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, X_train: Matrix, y_train: Matrix, X_val: Optional[Matrix] = None,
              y_val: Optional[Matrix] = None, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.01, show_progress: bool = True,
              hard_mining: bool = False) -> Dict[str, Any]:
        """
        Entrena la red durante múltiples épocas
        
        Complejidad temporal: O(epochs * num_samples * (sum(I_i*O_i) / batch_size))
        Complejidad espacial: O(max(num_samples, sum(I_i*O_i)))
        
        Retorna: diccionario con histórico de entrenamiento
        """
        results = {
            'losses': [],
            'accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'epoch_times': [],
            'total_time': 0
        }
        
        start_total = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Entrenar época
            if hard_mining and epoch > 0:
                # Seleccionar muestras difíciles
                train_loss, train_acc = self._train_epoch_with_hard_mining(
                    X_train, y_train, batch_size, learning_rate
                )
            else:
                train_loss, train_acc = self.train_epoch(
                    X_train, y_train, batch_size, learning_rate
                )
            
            epoch_time = time.time() - epoch_start
            
            results['losses'].append(train_loss)
            results['accuracies'].append(train_acc)
            results['epoch_times'].append(epoch_time)
            
            # Validación
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = self.compute_loss(val_predictions, y_val)
                val_acc = self.compute_accuracy(val_predictions, y_val)
                
                results['val_losses'].append(val_loss)
                results['val_accuracies'].append(val_acc)
            
            # Mostrar progreso
            if show_progress and (epoch + 1) % max(1, epochs // 20) == 0:
                msg = f"Época {epoch+1}/{epochs} | "
                msg += f"Pérdida: {train_loss:.6f} | Precisión: {train_acc:.4f} | "
                msg += f"Tiempo: {epoch_time:.3f}s"
                
                if X_val is not None:
                    msg += f" | Val Pérdida: {val_loss:.6f} | Val Precisión: {val_acc:.4f}"
                
                print(msg)
        
        results['total_time'] = time.time() - start_total
        
        return results
    
    def _train_epoch_with_hard_mining(self, X_train: Matrix, y_train: Matrix,
                                     batch_size: int, learning_rate: float) -> Tuple[float, float]:
        """
        Entrena una época usando Hard Negative Mining
        Selecciona las muestras más difíciles para entrenar
        
        Complejidad temporal: O(num_samples) + O(num_samples log k) para heap
        = O(num_samples log batch_size) en total
        
        Complejidad espacial: O(num_samples) para pérdidas + O(batch_size) para heap
        """
        # Forward pass en todo el dataset
        all_predictions = self.forward(X_train)
        
        # Calcular pérdida por muestra
        losses = []
        for i in range(X_train.rows):
            sample_loss = 0.0
            for j in range(y_train.cols):
                if y_train.data[i][j] > 0:
                    pred = max(1e-7, min(1.0 - 1e-7, all_predictions.data[i][j]))
                    sample_loss -= y_train.data[i][j] * math.log(pred)
            losses.append(sample_loss)
        
        # Usar Quickselect para encontrar top-k
        top_k_indices = quickselect_top_k(losses, batch_size)
        
        # Entrenar solo con muestras difíciles
        X_batch = Matrix.zeros(len(top_k_indices), X_train.cols)
        y_batch = Matrix.zeros(len(top_k_indices), y_train.cols)
        
        for i, idx in enumerate(top_k_indices):
            X_batch.data[i] = X_train.data[idx][:]
            y_batch.data[i] = y_train.data[idx][:]
        
        # Forward y backward con muestras difíciles
        predictions = self.forward(X_batch)
        batch_loss = self.compute_loss(predictions, y_batch)
        batch_accuracy = self.compute_accuracy(predictions, y_batch)
        
        dLoss = softmax_derivative(predictions, y_batch)
        self.backward(dLoss, learning_rate)
        
        return batch_loss, batch_accuracy
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Hace predicciones
        Complejidad: O(sum(B*I_i*O_i))
        """
        return self.forward(X)
    
    def get_total_params(self) -> int:
        """
        Calcula número total de parámetros
        Complejidad: O(num_layers)
        """
        total = 0
        for layer in self.layers:
            total += layer.input_size * layer.output_size + layer.output_size
        return total
    
    def save_weights(self, filepath: str) -> bool:
        """
        Guarda los pesos y sesgos de la red en un archivo JSON
        
        Complejidad temporal: O(total_parameters) para serializar
        Complejidad espacial: O(total_parameters) para estructura de datos
        
        Retorna: True si se guardó exitosamente, False si no
        """
        try:
            weights_data = {
                'layer_weights': [],
                'layer_biases': []
            }
            
            for layer in self.layers:
                # Guardar pesos
                weights_data['layer_weights'].append(layer.W.data)
                # Guardar sesgos
                weights_data['layer_biases'].append(layer.b.data)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(weights_data, f)
            
            return True
        except Exception as e:
            print(f"Error guardando pesos: {e}")
            return False
    
    def load_weights(self, filepath: str) -> bool:
        """
        Carga los pesos y sesgos de la red desde un archivo JSON
        
        Complejidad temporal: O(total_parameters) para deserializar
        Complejidad espacial: O(total_parameters)
        
        Retorna: True si se cargó exitosamente, False si no
        """
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                weights_data = json.load(f)
            
            # Cargar pesos y sesgos
            for i, layer in enumerate(self.layers):
                if i < len(weights_data['layer_weights']):
                    layer.W.data = weights_data['layer_weights'][i]
                if i < len(weights_data['layer_biases']):
                    layer.b.data = weights_data['layer_biases'][i]
            
            return True
        except Exception as e:
            print(f"Error cargando pesos: {e}")
            return False
    
    def save_checkpoint(self, filepath: str, additional_data: Dict[str, Any] = None) -> bool:
        """
        Guarda un checkpoint completo incluyendo pesos y metadatos
        
        Complejidad: O(total_parameters)
        
        Retorna: True si se guardó exitosamente, False si no
        """
        try:
            checkpoint = {
                'weights': [],
                'biases': [],
                'metadata': additional_data or {}
            }
            
            for layer in self.layers:
                checkpoint['weights'].append(layer.W.data)
                checkpoint['biases'].append(layer.b.data)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f)
            
            return True
        except Exception as e:
            print(f"Error guardando checkpoint: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Carga un checkpoint completo
        
        Complejidad: O(total_parameters)
        
        Retorna: (éxito, metadatos)
        """
        try:
            if not os.path.exists(filepath):
                return False, {}
            
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            
            for i, layer in enumerate(self.layers):
                if i < len(checkpoint['weights']):
                    layer.W.data = checkpoint['weights'][i]
                if i < len(checkpoint['biases']):
                    layer.b.data = checkpoint['biases'][i]
            
            return True, checkpoint.get('metadata', {})
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")
            return False, {}
