"""
Analizador de Complejidad Teorica del MLP.
Implementado desde cero para el Proyecto Final ADA.

Calcula:
- Numero total de parametros
- Costo de operaciones en forward y backward
- Complejidad espacial
- Analisis por capa

Sin uso de NumPy ni frameworks externos.
"""

from typing import List, Dict, Any


class ComplexityAnalyzer:
    """
    Clase estatica para analizar la complejidad teorica de un MLP.
    """

    @staticmethod
    def theoretical_complexity_analysis(
            layer_sizes: List[int],
            batch_size: int,
            num_epochs: int,
            num_training_samples: int) -> Dict[str, Any]:
        """
        Calcula la complejidad teorica temporal y espacial del MLP.

        Modelo de costo:
        - Forward por capa i: O(B * I_i * O_i)  multiplicaciones
        - Backward por capa i: ~1.5x forward  (gradientes + actualizacion)
        - Total por epoca: O((N/B) * B * sum(I_i * O_i)) = O(N * sum(I_i * O_i))

        Donde:
        - B = batch_size
        - N = num_training_samples
        - I_i = input_size de capa i
        - O_i = output_size de capa i
        - E = num_epochs

        Complejidad espacial:
        - Pesos: sum(I_i * O_i + O_i)   (W y b de cada capa)
        - Activaciones en cache: B * max(I_i) * num_layers
        - Cache de backprop: igual que activaciones

        Parametros:
        - layer_sizes: lista con el tamano de cada capa [in, h1, h2, ..., out]
        - batch_size: tamano del mini-lote (B)
        - num_epochs: numero de epocas de entrenamiento (E)
        - num_training_samples: numero de muestras de entrenamiento (N)

        Retorna:
        - Diccionario con todas las metricas de complejidad
        """
        num_layers = len(layer_sizes) - 1

        # ---- Parametros ----
        total_params = 0
        layer_params = []
        for i in range(num_layers):
            W_params = layer_sizes[i] * layer_sizes[i + 1]
            b_params = layer_sizes[i + 1]
            params = W_params + b_params
            layer_params.append(params)
            total_params += params

        # ---- Costo por capa (FLOPS por muestra) ----
        # Forward: multiplicacion matricial I_i x O_i
        forward_flops_per_sample = sum(
            layer_sizes[i] * layer_sizes[i + 1]
            for i in range(num_layers)
        )

        # Backward: ~1.5x el forward (gradientes de W, b y propagacion)
        backward_flops_per_sample = int(1.5 * forward_flops_per_sample)

        total_flops_per_sample = forward_flops_per_sample + backward_flops_per_sample

        # ---- Costo por lote ----
        forward_cost_per_batch = batch_size * forward_flops_per_sample
        backward_cost_per_batch = batch_size * backward_flops_per_sample

        # ---- Costo por epoca ----
        num_batches_per_epoch = max(1, num_training_samples // batch_size)
        cost_per_epoch = num_batches_per_epoch * (
            forward_cost_per_batch + backward_cost_per_batch
        )

        # ---- Costo total (E epocas) ----
        total_time_operations = num_epochs * cost_per_epoch

        # ---- Complejidad espacial ----
        # Pesos y sesgos
        params_space = total_params

        # Cache de activaciones: B * max(capa) * num_layers
        max_layer_size = max(layer_sizes)
        activation_space = batch_size * max_layer_size * num_layers

        # Cache de backprop (igual que activaciones)
        cache_space = activation_space

        total_space_complexity = params_space + activation_space + cache_space

        # ---- Analisis por capa ----
        layer_analysis = []
        for i in range(num_layers):
            I = layer_sizes[i]
            O = layer_sizes[i + 1]
            layer_analysis.append({
                'layer': i + 1,
                'input_size': I,
                'output_size': O,
                'params': layer_params[i],
                'forward_ops': I * O,
                'backward_ops': int(1.5 * I * O)
            })

        return {
            # Parametros del modelo
            'total_parameters': total_params,
            'layer_params': layer_params,
            'layer_analysis': layer_analysis,

            # Complejidad temporal
            'forward_flops_per_sample': forward_flops_per_sample,
            'backward_flops_per_sample': backward_flops_per_sample,
            'forward_cost_per_batch': forward_cost_per_batch,
            'backward_cost_per_batch': backward_cost_per_batch,
            'cost_per_epoch': cost_per_epoch,
            'total_time_operations': total_time_operations,
            'num_batches_per_epoch': num_batches_per_epoch,

            # Complejidad espacial
            'params_space': params_space,
            'activation_space': activation_space,
            'cache_space': cache_space,
            'total_space_complexity': total_space_complexity,

            # Notacion Big-O (simbolica)
            'big_o_time': 'O(E * N/B * B * sum(I_i * O_i))',
            'big_theta_time': 'Theta(E * N * sum(I_i * O_i))',
            'big_o_space': 'O(sum(I_i * O_i) + B * max(I_i))',

            # Configuracion
            'layer_sizes': layer_sizes,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_training_samples': num_training_samples
        }

    @staticmethod
    def print_analysis(complexity: dict) -> None:
        """Imprime el analisis de complejidad de forma legible."""
        print("\n" + "=" * 70)
        print("  ANALISIS DE COMPLEJIDAD TEORICA DEL MLP")
        print("=" * 70)

        ls = complexity['layer_sizes']
        print(f"\n  Arquitectura: {' -> '.join(map(str, ls))}")
        print(f"  Parametros totales: {complexity['total_parameters']:,}")

        print("\n  Por capa:")
        print(f"  {'Capa':<8} {'Entrada':<10} {'Salida':<10} {'Params':<12} {'Ops Forward':<14}")
        print("  " + "-" * 60)
        for la in complexity['layer_analysis']:
            print(f"  {la['layer']:<8} {la['input_size']:<10} {la['output_size']:<10} "
                  f"{la['params']:<12,} {la['forward_ops']:<14,}")

        print(f"\n  COMPLEJIDAD TEMPORAL:")
        print(f"    Forward/batch:    {complexity['forward_cost_per_batch']:,} ops")
        print(f"    Backward/batch:   {complexity['backward_cost_per_batch']:,} ops")
        print(f"    Por epoca:        {complexity['cost_per_epoch']:,} ops")
        print(f"    Total:            {complexity['total_time_operations']:,} ops")
        print(f"    Big-O:  {complexity['big_o_time']}")
        print(f"    Theta:  {complexity['big_theta_time']}")

        print(f"\n  COMPLEJIDAD ESPACIAL:")
        print(f"    Pesos + sesgos:   {complexity['params_space']:,} floats")
        print(f"    Cache activac.:   {complexity['activation_space']:,} floats")
        print(f"    Cache backprop:   {complexity['cache_space']:,} floats")
        print(f"    Total:            {complexity['total_space_complexity']:,} floats")
        print(f"    En bytes (f32):   ~{complexity['total_space_complexity']*4/1024:.1f} KB")
        print(f"    Big-O:  {complexity['big_o_space']}")
