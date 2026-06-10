"""
Script principal de entrenamiento - PROYECTO FINAL ADA
Red Neuronal Multicapa (MLP) entrenada desde cero

Implementa las cuatro fases del proyecto:
  Fase 1 (RA1): MLP minimo, complejidad teorica, experimentos de escalado B/h/E/N
  Fase 2 (RA2): Quickselect (mediana/top-k), heap vs sort, Metodo Maestro
  Fase 3 (RA3): Cola de lotes, hash de perdidas, BST para poda
  Fase 4:       Baseline k-NN vs MLP, Heapsort vs Quicksort, gradient check

Restricciones:
  - Sin frameworks de alto nivel (solo Python estandar + modulos propios)
  - Permitido NumPy solo en analizar.py (modulo auxiliar externo)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import json
import random

from src.mlp import MLP
from src.dataset import DatasetGenerator
from src.complexity_analyzer import ComplexityAnalyzer
from src.data_structures import (
    quickselect_median, heap_top_k_vs_sort,
    imprimir_analisis_ra2, imprimir_analisis_ra3,
    podar_pesos_bst, construir_bst_desde_modelo
)
from src.sorting import comparar_heapsort_vs_quicksort, imprimir_comparativa
from src.knn import KNNClassifier, comparar_knn_vs_mlp, imprimir_comparativa_knn
from src.gradient_check import gradient_check
from src.experiments import ejecutar_todos_experimentos


# ============================================================
# CONFIGURACION GLOBAL
# ============================================================

WEIGHTS_CHECKPOINT_PATH = 'model_state/weights_checkpoint.json'
TRAINING_HISTORY_PATH   = 'model_state/training_history.json'


# ============================================================
# UTILIDADES DE IMPRESION
# ============================================================

def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100 + "\n")


def barra_progreso(epoch, total, loss, acc, elapsed, eta):
    prog   = epoch / total
    filled = int(40 * prog)
    barra  = '#' * filled + '.' * (40 - filled)
    print(f"[{barra}] {prog*100:5.1f}% | "
          f"Epoca {epoch:4d}/{total} | "
          f"Perdida: {loss:.6f} | "
          f"Precision: {acc:.4f} | "
          f"Tiempo: {elapsed:6.1f}s | "
          f"ETA: {eta:6.1f}s")


# ============================================================
# PERSISTENCIA
# ============================================================

def cargar_historial():
    if os.path.exists(TRAINING_HISTORY_PATH):
        try:
            with open(TRAINING_HISTORY_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'total_executions': 0,
        'total_epochs_trained': 0,
        'all_losses': [],
        'all_accuracies': [],
        'execution_history': []
    }


def guardar_historial(historial):
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    with open(TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(historial, f, indent=2)


# ============================================================
# FASE 1 (RA1): ANALISIS TEORICO DE COMPLEJIDAD
# ============================================================

def fase1_complejidad(layer_sizes, batch_size, epochs, num_samples):
    """
    RA1: Derivacion de complejidad temporal/espacial del MLP.
    Incluye notacion Big-O, Big-Omega y Big-Theta.
    """
    subheader("FASE 1 (RA1): ANALISIS TEORICO DE COMPLEJIDAD")

    complexity = ComplexityAnalyzer.theoretical_complexity_analysis(
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        num_epochs=epochs,
        num_training_samples=num_samples
    )

    print(f"Arquitectura de la Red: {' -> '.join(map(str, layer_sizes))}")
    print(f"Parametros totales: {complexity['total_parameters']:,}")
    print(f"Epocas: {epochs} | Batch size: {batch_size} | Muestras: {num_samples}")

    print("\nComplejidad TEMPORAL:")
    print(f"  Forward pass por lote:  O(B * sum(I_i * O_i)) = O({complexity['forward_cost_per_batch']:,})")
    print(f"  Backward pass por lote: O({int(complexity['backward_cost_per_batch']):,}) ~ 1.5x forward")
    print(f"  Costo por epoca:        O({int(complexity['cost_per_epoch']):,})")
    print(f"  Total ({epochs} epocas):   O({int(complexity['total_time_operations']):,})")
    print(f"\n  Notacion simplificada: Theta(E * N * sum(I_i * O_i) / B)")
    print(f"  Big-O = Big-Omega -> Cota ajustada Theta (el algoritmo")
    print(f"  SIEMPRE debe procesar todos los datos en todos los lotes)")

    print("\nComplejidad ESPACIAL:")
    print(f"  Pesos + gradientes:     O({complexity['params_space']:,})")
    print(f"  Activaciones en cache:  O({complexity['activation_space']:,}) = O(B * max(I_i))")
    print(f"  Cache backprop:         O({complexity['cache_space']:,})")
    print(f"  Total:                  O({complexity['total_space_complexity']:,} floats)")
    print(f"  En bytes (float32):     ~{complexity['total_space_complexity']*4/1024:.1f} KB")
    print(f"\n  Big-O espacial:   O(sum(I_i*O_i) + B*max(I_i))")
    print(f"  Big-Omega espacial: Omega(sum(I_i*O_i))")
    print(f"  No son iguales -> depende de si se guarda cache de activaciones")

    return complexity


# ============================================================
# FASE 1 (RA1): EXPERIMENTOS DE ESCALADO
# ============================================================

def fase1_experimentos(layer_sizes, n_base=500):
    """RA1: Experimentos empiricos variando B, h, E, N."""
    subheader("FASE 1 (RA1): EXPERIMENTOS DE ESCALADO EMPIRICO")
    print("  Valida que las derivaciones teoricas coinciden con mediciones reales\n")
    resultados = ejecutar_todos_experimentos(layer_sizes, N_base=n_base)
    return resultados


# ============================================================
# FASE 2 (RA2): SELECCION Y ORDENAMIENTO
# ============================================================

def fase2_seleccion_ordenamiento():
    """
    RA2: Demostracion de Quickselect (mediana), heap vs sort, Metodo Maestro.
    """
    subheader("FASE 2 (RA2): SELECCION Y ORDENAMIENTO")

    # Demostracion de Quickselect para mediana
    print("  [Quickselect - Mediana]")
    datos_ejemplo = [random.uniform(0, 10) for _ in range(1000)]
    mediana_qs = quickselect_median(datos_ejemplo)
    datos_ord  = sorted(datos_ejemplo)
    mediana_ord = datos_ord[len(datos_ord) // 2]
    error = abs(mediana_qs - mediana_ord)
    print(f"    n=1000 elementos | Mediana Quickselect: {mediana_qs:.4f}")
    print(f"    Mediana por sort: {mediana_ord:.4f} | Error: {error:.6f}")
    print(f"    Quickselect O(n) vs sort O(n log n) para solo la mediana")

    # Comparativa heap vs sort
    imprimir_analisis_ra2(n_values=[200, 500, 1000, 3000], k_values=[10, 30])


# ============================================================
# FASE 3 (RA3): ESTRUCTURAS DE DATOS INTEGRADAS
# ============================================================

def fase3_estructuras(model):
    """RA3: BST para poda, hash de perdidas, cola de lotes."""
    imprimir_analisis_ra3(model)

    subheader("FASE 3 (RA3): PODA DE PESOS CON BST")
    for umbral in [0.001, 0.005, 0.01]:
        # Hacemos copia para no afectar el modelo entrenado
        import copy
        model_copy = copy.deepcopy(model)
        stats = podar_pesos_bst(model_copy, umbral=umbral)
        print(f"  Umbral {umbral:.3f}: {stats['pesos_podados']:,}/{stats['total_pesos']:,} "
              f"pesos podados ({stats['ratio_poda']*100:.1f}%)")


# ============================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar_modelo(model, X_train, y_train, X_val, y_val,
                    epochs, batch_size, learning_rate,
                    titulo="ENTRENAMIENTO"):
    """Entrena el modelo mostrando progreso por epoca."""
    print(f"\n  Arquitectura: Input({X_train.cols}) -> Hidden(64) -> Hidden(64) -> Output({y_train.cols})")
    print(f"  Muestras train: {X_train.rows} | Validacion: {X_val.rows}")
    print(f"  Batch size: {batch_size} | LR: {learning_rate} | Epocas: {epochs}")
    print(f"  Parametros totales: {model.get_total_params():,}\n")

    resultados = {
        'losses': [], 'accuracies': [],
        'val_losses': [], 'val_accuracies': [],
        'epoch_times': [], 'total_time': 0,
        'epochs_in_this_run': epochs
    }

    start_total = time.time()
    tiempos_recientes = []

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = model.train_epoch(
            X_train, y_train, batch_size, learning_rate
        )

        epoch_time = time.time() - t0
        tiempos_recientes.append(epoch_time)
        if len(tiempos_recientes) > 10:
            tiempos_recientes = tiempos_recientes[-10:]

        avg_t = sum(tiempos_recientes) / len(tiempos_recientes)
        eta = avg_t * (epochs - epoch - 1)
        elapsed = time.time() - start_total

        resultados['losses'].append(train_loss)
        resultados['accuracies'].append(train_acc)
        resultados['epoch_times'].append(epoch_time)

        val_pred = model.forward(X_val)
        val_loss = model.compute_loss(val_pred, y_val)
        val_acc  = model.compute_accuracy(val_pred, y_val)
        resultados['val_losses'].append(val_loss)
        resultados['val_accuracies'].append(val_acc)

        barra_progreso(epoch + 1, epochs, train_loss, train_acc, elapsed, eta)

        # Detalle cada 25 epocas o en la primera/ultima
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 25 == 0:
            print(f"\n  L- Epoca {epoch+1:4d}: "
                  f"Train={train_loss:.6f}/{train_acc:.4f} | "
                  f"Val={val_loss:.6f}/{val_acc:.4f} | "
                  f"Tiempo={epoch_time:.3f}s\n")

    resultados['total_time'] = time.time() - start_total
    return resultados


# ============================================================
# RESUMEN DE ENTRENAMIENTO
# ============================================================

def imprimir_resumen(resultados, historial_completo):
    subheader("4. RESULTADOS FINALES DEL ENTRENAMIENTO")

    epocas_totales = historial_completo['total_epochs_trained']
    print(f"  * Epocas totales entrenadas hasta ahora: {epocas_totales}")
    print(f"  * Epocas en esta ejecucion: {resultados['epochs_in_this_run']}")
    print(f"  * Perdida final: {resultados['losses'][-1]:.6f}")
    print(f"  * Precision final: {resultados['accuracies'][-1]:.4f}")
    print(f"  * Tiempo total: {resultados['total_time']:.2f}s")
    print(f"  * Tiempo promedio por epoca: "
          f"{sum(resultados['epoch_times'])/len(resultados['epoch_times']):.4f}s")

    max_acc     = max(resultados['accuracies'])
    max_val_acc = max(resultados['val_accuracies'])
    print(f"\n  Mejor precision entrenamiento: {max_acc:.4f}")
    print(f"  Mejor precision validacion:    {max_val_acc:.4f}")

    # Detectar sobreajuste
    if len(resultados['val_losses']) > 5:
        train_avg = sum(resultados['losses'][-5:]) / 5
        val_avg   = sum(resultados['val_losses'][-5:]) / 5
        ratio     = val_avg / train_avg if train_avg > 0 else 1.0
        print(f"\n  Ratio Val/Train (ultimas 5 epocas): {ratio:.4f}")
        if ratio > 1.1:
            print("  [!] Posible sobreajuste detectado (ratio > 1.1)")
        else:
            print("  [OK] Modelo generaliza bien (ratio <= 1.1)")


# ============================================================
# MAIN
# ============================================================

def main():
    header("PROYECTO FINAL ADA - MLP DESDE CERO - ANALISIS DE COMPLEJIDAD")

    # --------------------------------------------------------
    # CONFIGURACION
    # --------------------------------------------------------
    EPOCHS_PER_RUN  = 100 # Reducido para demostracion rapida; aumentar a 100 para mejor acc
    BATCH_SIZE      = 32
    LEARNING_RATE   = 0.01
    HIDDEN_SIZE     = 4
    NUM_SAMPLES     = 800  # Reducido para que los experimentos de escalado no tarden demasiado
    NUM_FEATURES    = 20   # Caracteristicas (no 784 para que sea rapido sin NumPy)
    NUM_CLASSES     = 10

    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES]

    historial = cargar_historial()
    model_exists = os.path.exists(WEIGHTS_CHECKPOINT_PATH)

    # --------------------------------------------------------
    # FASE 1 (RA1): COMPLEJIDAD TEORICA
    # --------------------------------------------------------
    header("FASE 1 (RA1): COMPLEJIDAD TEORICA")
    complexity = fase1_complejidad(
        layer_sizes, BATCH_SIZE, EPOCHS_PER_RUN, NUM_SAMPLES
    )

    if historial['total_executions'] == 0:
        os.makedirs('model_state', exist_ok=True)
        with open('model_state/complexity_analysis.json', 'w') as f:
            json.dump(complexity, f, indent=2)

    # --------------------------------------------------------
    # FASE 1 (RA1): EXPERIMENTOS DE ESCALADO
    # --------------------------------------------------------
    header("FASE 1 (RA1): EXPERIMENTOS DE ESCALADO (B, h, E, N)")
    fase1_experimentos(layer_sizes, n_base=NUM_SAMPLES)

    # --------------------------------------------------------
    # FASE 2 (RA2): SELECCION Y ORDENAMIENTO
    # --------------------------------------------------------
    header("FASE 2 (RA2): SELECCION Y ORDENAMIENTO")
    fase2_seleccion_ordenamiento()

    # --------------------------------------------------------
    # PREPARACION DEL DATASET
    # --------------------------------------------------------
    header("PREPARACION DEL DATASET")

    print("  Generando dataset sintetico de clasificacion...")
    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        seed=42
    )

    print("  Normalizando datos (Z-score)...")
    X = DatasetGenerator.normalize_dataset(X)

    print("  Dividiendo: 80% entrenamiento / 20% validacion...")
    X_train, y_train, X_val, y_val = DatasetGenerator.split_dataset(
        X, y, train_ratio=0.8, seed=42
    )

    print(f"\n  Dataset listo:")
    print(f"    Muestras entrenamiento: {X_train.rows}")
    print(f"    Muestras validacion:    {X_val.rows}")
    print(f"    Caracteristicas:        {X_train.cols}")
    print(f"    Clases:                 {y_train.cols}")

    # --------------------------------------------------------
    # CREACION / CARGA DEL MODELO
    # --------------------------------------------------------
    header("CREACION / CARGA DEL MODELO")

    model = MLP(
        layer_sizes=layer_sizes,
        activations=['relu', 'linear'],
        loss_function='cross_entropy'
    )

    print(f"  Capas: {len(model.layers)} | Parametros: {model.get_total_params():,}")

    if model_exists and model.load_weights(WEIGHTS_CHECKPOINT_PATH):
        print(f"  Pesos cargados desde ejecucion anterior (aprendizaje progresivo)")
    else:
        print(f"  Primera ejecucion: pesos inicializados con distribucion He")

    # --------------------------------------------------------
    # VERIFICACION DE GRADIENTES (Correctitud)
    # --------------------------------------------------------
    header("VERIFICACION NUMERICA DE GRADIENTES (Correctitud)")
    print("  Usando un mini-batch pequeno para la verificacion...")

    # Tomar un lote pequeno para gradient check (es costoso)
    X_gc = X_train
    y_gc = y_train
    if X_train.rows > 20:
        from src.matrix import Matrix
        X_gc = Matrix(20, X_train.cols, [X_train.data[i][:] for i in range(20)])
        y_gc = Matrix(20, y_train.cols, [y_train.data[i][:] for i in range(20)])

    gc_results = gradient_check(model, X_gc, y_gc,
                                 epsilon=1e-5, num_checks=15, verbose=True)

    # --------------------------------------------------------
    # ENTRENAMIENTO PRINCIPAL
    # --------------------------------------------------------
    header(f"ENTRENAMIENTO - Ejecucion #{historial['total_executions'] + 1}")

    subheader("3. INFORMACION POR CADA EPOCA")
    resultados = entrenar_modelo(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS_PER_RUN,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        titulo=f"#{historial['total_executions'] + 1}"
    )

    initial_loss = resultados['losses'][0]
    final_loss   = resultados['losses'][-1]
    total_epocas = historial['total_epochs_trained'] + EPOCHS_PER_RUN

    # Actualizar historial con el registro actual para imprimir_resumen
    historial['total_epochs_trained'] = total_epocas

    imprimir_resumen(resultados, historial)

    # --------------------------------------------------------
    # FASE 3 (RA3): ESTRUCTURAS DE DATOS
    # --------------------------------------------------------
    header("FASE 3 (RA3): ESTRUCTURAS DE DATOS INTEGRADAS")
    fase3_estructuras(model)

    # --------------------------------------------------------
    # EVALUACION FINAL DEL MLP
    # --------------------------------------------------------
    header("EVALUACION FINAL DEL MLP")
    final_preds  = model.forward(X_val)
    final_acc    = model.compute_accuracy(final_preds, y_val)
    final_loss_v = model.compute_loss(final_preds, y_val)
    print(f"\n  Precision en validacion: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  Perdida en validacion:   {final_loss_v:.6f}")

    objetivo = 0.85
    if final_acc >= objetivo:
        print(f"\n  [OK] Precision >= {objetivo*100:.0f}% (criterio del enunciado alcanzado)")
    else:
        print(f"\n  [!] Precision < {objetivo*100:.0f}%. Ejecutar mas veces para mejorar.")

    # --------------------------------------------------------
    # FASE 4: BASELINE k-NN vs MLP
    # --------------------------------------------------------
    header("FASE 4: BASELINE k-NN vs MLP")
    print("  Evaluando k-NN con k = {1, 3, 5, 7, 11}...")
    print("  (Puede tardar algunos segundos por la complejidad O(M*N*d) del k-NN)\n")

    knn_results = comparar_knn_vs_mlp(
        X_train, y_train, X_val, y_val,
        mlp_model=model,
        k_values=[1, 3, 5, 7, 11]
    )
    imprimir_comparativa_knn(knn_results)

    # --------------------------------------------------------
    # FASE 4: HEAPSORT vs QUICKSORT
    # --------------------------------------------------------
    header("FASE 4: COMPARATIVA HEAPSORT vs QUICKSORT")
    sort_results = comparar_heapsort_vs_quicksort(
        sizes=[100, 500, 1000, 2000, 5000],
        repetitions=3
    )
    imprimir_comparativa(sort_results)

    # --------------------------------------------------------
    # PERSISTENCIA DEL MODELO
    # --------------------------------------------------------
    header("PERSISTENCIA DEL MODELO")
    if model.save_weights(WEIGHTS_CHECKPOINT_PATH):
        print(f"  Pesos guardados en: {WEIGHTS_CHECKPOINT_PATH}")
        print(f"  La proxima ejecucion continuara desde estos pesos")
    else:
        print(f"  [!] Error al guardar pesos")

    # Actualizar historial
    registro = {
        'execution_number': historial['total_executions'] + 1,
        'epochs_in_run': EPOCHS_PER_RUN,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'final_accuracy': final_acc,
        'val_final_accuracy': resultados['val_accuracies'][-1],
        'total_time': resultados['total_time'],
        'gradient_check_passed': gc_results['passed'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    historial['total_executions']    += 1
    historial['execution_history'].append(registro)
    historial['all_losses'].extend(resultados['losses'])
    historial['all_accuracies'].extend(resultados['accuracies'])

    guardar_historial(historial)

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------
    header("RESUMEN GENERAL DE PROGRESION")
    print(f"  Ejecuciones completadas:      {historial['total_executions']}")
    print(f"  Epocas totales entrenadas:    {historial['total_epochs_trained']}")
    print(f"  Gradient check:               {'PASO' if gc_results['passed'] else 'FALLO'} "
          f"(error max = {gc_results['max_error']:.2e})")
    print(f"  Precision final MLP:          {final_acc:.4f}")
    print(f"  Mejor k-NN precision:         "
          f"{max(knn_results['knn_accuracies']):.4f} "
          f"(k={knn_results['k_values'][knn_results['knn_accuracies'].index(max(knn_results['knn_accuracies']))]})")

    if len(historial['execution_history']) > 1:
        prev = historial['execution_history'][-2]
        curr = historial['execution_history'][-1]
        delta_loss = prev['final_loss'] - curr['final_loss']
        delta_acc  = curr['final_accuracy'] - prev['final_accuracy']
        print(f"\n  Comparacion con ejecucion anterior:")
        print(f"    Cambio en perdida:    {delta_loss:+.6f} "
              f"({'mejoro' if delta_loss > 0 else 'empeoro'})")
        print(f"    Cambio en precision:  {delta_acc:+.4f} "
              f"({'mejoro' if delta_acc > 0 else 'empeoro'})")

    print(f"\n  Para continuar el entrenamiento, ejecuta el programa nuevamente.")
    print(f"  Cada ejecucion agrega {EPOCHS_PER_RUN} epocas adicionales.")

    print("\n" + "=" * 100)
    print("  PROYECTO FINAL ADA - COMPLETADO EXITOSAMENTE")
    print("  Fases RA1, RA2, RA3 y Fase 4 ejecutadas correctamente")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
