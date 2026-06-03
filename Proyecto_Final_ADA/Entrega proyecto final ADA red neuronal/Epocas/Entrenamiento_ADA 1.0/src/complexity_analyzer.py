"""
Análisis teórico y empírico de complejidad
Medición de tiempo y memoria durante el entrenamiento
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Any

# Agregar parent directory al path si es necesario
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlp import MLP
from src.matrix import Matrix

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class ComplexityAnalyzer:
    """
    Analizador de complejidad temporal y espacial
    """
    
    @staticmethod
    def theoretical_complexity_analysis(layer_sizes: List[int], 
                                       batch_size: int,
                                       num_epochs: int,
                                       num_training_samples: int) -> Dict[str, Any]:
        """
        Calcula la complejidad teórica de la red
        
        Parámetros:
        - layer_sizes: [input, hidden1, hidden2, ..., output]
        - batch_size: B
        - num_epochs: E
        - num_training_samples: N
        """
        
        # Calular parámetros
        total_params = 0
        weight_matrices = []
        
        for i in range(len(layer_sizes) - 1):
            params = layer_sizes[i] * layer_sizes[i+1]
            total_params += params + layer_sizes[i+1]  # +bias
            weight_matrices.append((layer_sizes[i], layer_sizes[i+1]))
        
        num_batches_per_epoch = (num_training_samples + batch_size - 1) // batch_size
        
        # Análisis de complejidad temporal
        # Forward pass: O(sum(B * I_i * O_i))
        forward_cost_per_batch = 0
        for I, O in weight_matrices:
            forward_cost_per_batch += batch_size * I * O
        
        # Backward pass: similar a forward
        backward_cost_per_batch = forward_cost_per_batch * 1.5  # aproximadamente 1.5x forward
        
        # Costo por época
        cost_per_epoch = num_batches_per_epoch * (forward_cost_per_batch + backward_cost_per_batch)
        
        # Costo total
        total_time_complexity = num_epochs * cost_per_epoch
        
        # Análisis de complejidad espacial
        # Almacenar pesos y gradientes
        params_space = total_params * 2  # pesos + gradientes
        
        # Almacenar activaciones por capa
        activation_space = batch_size * max(layer_sizes)
        
        # Cache para backward pass
        cache_space = batch_size * max(layer_sizes)
        
        total_space_complexity = params_space + activation_space + cache_space
        
        return {
            'total_parameters': total_params,
            'weight_matrices': weight_matrices,
            'batches_per_epoch': num_batches_per_epoch,
            'forward_cost_per_batch': forward_cost_per_batch,
            'backward_cost_per_batch': backward_cost_per_batch,
            'cost_per_epoch': cost_per_epoch,
            'total_time_operations': total_time_complexity,
            'time_complexity_notation': f"O({num_epochs} * {num_batches_per_epoch} * B * (sum(I_i * O_i))) = O({total_time_complexity})",
            'params_space': params_space,
            'activation_space': activation_space,
            'cache_space': cache_space,
            'total_space_complexity': total_space_complexity,
            'space_complexity_notation': f"O(sum(I_i * O_i) + B * max(I_i)) = O({total_space_complexity})"
        }
    
    @staticmethod
    def measure_epoch_performance(model: MLP, X_batch: Matrix, y_batch: Matrix,
                                 batch_size: int, learning_rate: float,
                                 num_repetitions: int = 3) -> Dict[str, float]:
        """
        Mide el tiempo de ejecución de una época
        Realiza múltiples repeticiones para obtener promedios confiables
        
        Retorna: diccionario con tiempos medidos
        """
        times = []
        
        for _ in range(num_repetitions):
            start = time.perf_counter()
            model.train_epoch(X_batch, y_batch, batch_size, learning_rate)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    @staticmethod
    def memory_usage() -> float:
        """
        Obtiene el uso de memoria actual en MB
        Complejidad: O(1)
        """
        if HAS_PSUTIL:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        else:
            # Si psutil no está disponible, retornar 0
            return 0.0
    
    @staticmethod
    def analyze_scaling(layer_sizes: List[int],
                       num_samples_values: List[int],
                       batch_size: int = 32,
                       epochs: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analiza cómo escala el tiempo con el número de muestras
        
        Complejidad: O(sum(epochs * num_samples * param_count))
        """
        from src.dataset import DatasetGenerator
        
        results = {
            'samples': [],
            'times': [],
            'memory': [],
            'samples_per_second': []
        }
        
        for num_samples in num_samples_values:
            print(f"  Analizando con {num_samples} muestras...")
            
            # Generar datos
            X, y = DatasetGenerator.generate_synthetic_classification(
                num_samples=num_samples,
                num_features=layer_sizes[0],
                num_classes=layer_sizes[-1],
                seed=42
            )
            
            X, y, _, _ = DatasetGenerator.split_dataset(X, y, train_ratio=1.0)
            
            # Crear modelo
            model = MLP(layer_sizes, ['relu'] * (len(layer_sizes) - 2) + ['linear'])
            
            # Medir memoria inicial
            mem_before = ComplexityAnalyzer.memory_usage()
            
            # Medir tiempo de entrenamiento
            start = time.perf_counter()
            model.train(X, y, epochs=epochs, batch_size=batch_size,
                       learning_rate=0.01, show_progress=False)
            end = time.perf_counter()
            
            # Medir memoria final
            mem_after = ComplexityAnalyzer.memory_usage()
            
            elapsed_time = end - start
            samples_per_sec = num_samples * epochs / elapsed_time
            
            results['samples'].append(num_samples)
            results['times'].append(elapsed_time)
            results['memory'].append(mem_after - mem_before)
            results['samples_per_second'].append(samples_per_sec)
        
        return results
    
    @staticmethod
    def analyze_batch_size_effect(layer_sizes: List[int],
                                 num_samples: int,
                                 batch_sizes: List[int],
                                 epochs: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analiza el efecto del tamaño de lote en el tiempo de entrenamiento
        
        Expectativa teórica: tiempo ≈ O(num_samples / batch_size)
        (inversamente proporcional al tamaño del lote)
        """
        from src.dataset import DatasetGenerator
        
        results = {
            'batch_sizes': [],
            'times': [],
            'throughput': []  # samples/second
        }
        
        # Generar datos una sola vez
        X, y = DatasetGenerator.generate_synthetic_classification(
            num_samples=num_samples,
            num_features=layer_sizes[0],
            num_classes=layer_sizes[-1],
            seed=42
        )
        X, y, _, _ = DatasetGenerator.split_dataset(X, y, train_ratio=1.0)
        
        for batch_size in batch_sizes:
            print(f"  Analizando con batch_size={batch_size}...")
            
            # Crear modelo
            model = MLP(layer_sizes, ['relu'] * (len(layer_sizes) - 2) + ['linear'])
            
            # Medir tiempo
            start = time.perf_counter()
            model.train(X, y, epochs=epochs, batch_size=batch_size,
                       learning_rate=0.01, show_progress=False)
            end = time.perf_counter()
            
            elapsed_time = end - start
            throughput = (num_samples * epochs) / elapsed_time
            
            results['batch_sizes'].append(batch_size)
            results['times'].append(elapsed_time)
            results['throughput'].append(throughput)
        
        return results
    
    @staticmethod
    def analyze_hidden_layer_effect(input_size: int, output_size: int,
                                   hidden_sizes: List[int],
                                   num_samples: int = 1000,
                                   epochs: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analiza cómo afecta el tamaño de capas ocultas
        
        Expectativa teórica: tiempo ≈ O(sum(I_i * O_i))
        """
        from src.dataset import DatasetGenerator
        
        results = {
            'hidden_sizes': [],
            'times': [],
            'parameters': [],
            'time_per_param': []
        }
        
        # Generar datos una sola vez
        X, y = DatasetGenerator.generate_synthetic_classification(
            num_samples=num_samples,
            num_features=input_size,
            num_classes=output_size,
            seed=42
        )
        X, y, _, _ = DatasetGenerator.split_dataset(X, y, train_ratio=1.0)
        
        for hidden_size in hidden_sizes:
            print(f"  Analizando con hidden_size={hidden_size}...")
            
            # Crear modelo
            layer_sizes = [input_size, hidden_size, output_size]
            model = MLP(layer_sizes, ['relu', 'linear'])
            
            num_params = model.get_total_params()
            
            # Medir tiempo
            start = time.perf_counter()
            model.train(X, y, epochs=epochs, batch_size=32,
                       learning_rate=0.01, show_progress=False)
            end = time.perf_counter()
            
            elapsed_time = end - start
            time_per_param = elapsed_time / num_params
            
            results['hidden_sizes'].append(hidden_size)
            results['times'].append(elapsed_time)
            results['parameters'].append(num_params)
            results['time_per_param'].append(time_per_param)
        
        return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("  ANÁLISIS DETALLADO DE COMPLEJIDAD - RED NEURAL")
    print("="*80 + "\n")
    
    # Parámetros
    layer_sizes = [784, 128, 64, 10]
    batch_size = 32
    epochs = 1000
    num_samples = 10000
    
    # Análisis teórico
    complexity = ComplexityAnalyzer.theoretical_complexity_analysis(
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        num_epochs=epochs,
        num_training_samples=num_samples
    )
    
    # ============ INFORMACIÓN GENERAL ============
    print("📋 CONFIGURACIÓN DE LA RED")
    print("-" * 80)
    print(f"  Arquitectura: {' → '.join(map(str, layer_sizes))}")
    print(f"  Parámetros totales: {complexity['total_parameters']:,}")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Muestras de entrenamiento: {num_samples:,}")
    print(f"  Batches por época: {complexity['batches_per_epoch']}")
    
    # ============ BIG O (COTA SUPERIOR) ============
    print("\n" + "="*80)
    print("📈 ANÁLISIS BIG O (COTA SUPERIOR - PEOR CASO)")
    print("="*80)
    
    print("\n🕐 COMPLEJIDAD TEMPORAL (BIG O):")
    print("-" * 80)
    print(f"""
  Forward Pass por batch:  O(B × Σ(I_i × O_i))
                          = O({batch_size} × {complexity['forward_cost_per_batch']:,})
                          = O({complexity['forward_cost_per_batch']:,})
    
  Backward Pass por batch: O(1.5 × B × Σ(I_i × O_i))
                          = O({int(complexity['backward_cost_per_batch']):,})
    
  Una época:              O(⌈N/B⌉ × (forward + backward))
                          = O({complexity['batches_per_epoch']} × {int(complexity['forward_cost_per_batch'] + complexity['backward_cost_per_batch']):,})
                          = O({int(complexity['cost_per_epoch']):,})
    
  TOTAL ({epochs} épocas):  O(E × ⌈N/B⌉ × (forward + backward))
                          = O({epochs} × {int(complexity['cost_per_epoch']):,})
                          = O({int(complexity['total_time_operations']):,})
    """)
    
    print(f"  Notación simplificada: O(E × N × Σ(I_i × O_i) / B)")
    print(f"  Donde:")
    print(f"    E = Épocas = {epochs}")
    print(f"    N = Muestras = {num_samples:,}")
    print(f"    B = Batch size = {batch_size}")
    print(f"    Σ(I_i × O_i) = Suma de productos de capas = {sum(i*o for i, o in complexity['weight_matrices']):,}")
    
    print("\n💾 COMPLEJIDAD ESPACIAL (BIG O):")
    print("-" * 80)
    print(f"""
  Almacenamiento de pesos:     O(Σ(I_i × O_i))
                              = O({complexity['params_space']:,})
    
  Almacenamiento de gradientes: O(Σ(I_i × O_i))
                              = O({complexity['params_space']:,})
    
  Activaciones en caché:        O(B × max(I_i))
                              = O({batch_size} × {max(layer_sizes)})
                              = O({complexity['activation_space']:,})
    
  TOTAL:                        O(2×Σ(I_i × O_i) + B × max(I_i))
                              = O({complexity['total_space_complexity']:,})
    """)
    
    # ============ BIG OMEGA (COTA INFERIOR) ============
    print("\n" + "="*80)
    print("📉 ANÁLISIS BIG OMEGA (COTA INFERIOR - MEJOR CASO)")
    print("="*80)
    
    print("\n🕐 COMPLEJIDAD TEMPORAL (BIG OMEGA):")
    print("-" * 80)
    print(f"""
  Forward Pass por batch:  Ω(B × Σ(I_i × O_i))
                          = Ω({complexity['forward_cost_per_batch']:,})
    
  Backward Pass por batch: Ω(B × Σ(I_i × O_i))
                          = Ω({int(complexity['backward_cost_per_batch']):,})
    
  Una época:              Ω(⌈N/B⌉ × (forward + backward))
                          = Ω({int(complexity['cost_per_epoch']):,})
    
  TOTAL ({epochs} épocas):  Ω(E × ⌈N/B⌉ × 2 × B × Σ(I_i × O_i))
                          = Ω({int(complexity['total_time_operations']):,})
    """)
    
    print(f"  Explicación: Aunque el mejor caso teórico podría ser menor,")
    print(f"  en la práctica SIEMPRE debe procesar todos los datos")
    print(f"  Ω(E × N × Σ(I_i × O_i) / B)")
    
    print("\n💾 COMPLEJIDAD ESPACIAL (BIG OMEGA):")
    print("-" * 80)
    print(f"""
  Mínimo de memoria:        Ω(Σ(I_i × O_i))
                           = Ω({complexity['params_space']:,})
    
  Explicación: Se requieren almacenar TODOS los pesos
  sin excepción en cualquier implementación
    """)
    
    # ============ ANÁLISIS COMPARATIVO ============
    print("\n" + "="*80)
    print("⚖️  ANÁLISIS BIG O vs BIG OMEGA (Tight Bounds)")
    print("="*80)
    
    print("""
  COMPLEJIDAD TEMPORAL:
  ─────────────────────
    Big O (peor caso):    O(E × N × Σ(I_i × O_i) / B)
    Big Omega (mejor):    Ω(E × N × Σ(I_i × O_i) / B)
    
    ✓ Son IGUALES → El algoritmo tiene complejidad Θ(E × N × Σ(I_i × O_i) / B)
      (tight bound - cota ajustada)
    
    Interpretación:
    - El entrenamiento SIEMPRE toma el mismo orden de tiempo
    - No hay caminos de ejecución más rápidos o más lentos
    - El tiempo es proporcional a: épocas × datos × complejidad de red
    - Inversamente proporcional al tamaño de batch
    
  COMPLEJIDAD ESPACIAL:
  ────────────────────
    Big O (peor caso):    O(Σ(I_i × O_i) + B × max(I_i))
    Big Omega (mejor):    Ω(Σ(I_i × O_i))
    
    ✗ NO son iguales → El algoritmo tiene complejidad entre Ω y O
    
    Explicación:
    - Mínimo: Necesitamos almacenar todos los pesos (Ω)
    - Máximo: Además almacenamos activaciones en caché (O)
    - El overhead de caché depende del batch size
    
  CONCLUSIÓN:
  ──────────
    Tiempo:   Θ(E × N × Σ(I_i × O_i) / B) - Tight bound
    Espacio:  O(Σ(I_i × O_i) + B × max(I_i)) - Can be optimized
    """)
    
    # ============ NÚMEROS CONCRETOS ============
    print("\n" + "="*80)
    print("🔢 NÚMEROS CONCRETOS PARA ESTA RED")
    print("="*80)
    print(f"""
  Operaciones totales:     {int(complexity['total_time_operations']):,} operaciones
  Memoria total:           {complexity['total_space_complexity']:,} floats
  
  En tiempo:
  - Estimado: ~{int(complexity['total_time_operations'] / 1e9):.2f} mil millones de operaciones
  - En procesador moderno (~1 GFlops): ~{int(complexity['total_time_operations'] / 1e9):.0f} segundos
  
  En memoria:
  - Floats: {complexity['total_space_complexity']:,}
  - Bytes: {complexity['total_space_complexity'] * 4:,} (32-bit floats)
  - MB: {complexity['total_space_complexity'] * 4 / (1024*1024):.2f}
    """)
    
    print("\n✓ Análisis completado correctamente")
