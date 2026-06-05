"""
Script principal de entrenamiento
Entrena la red neuronal con persistencia de pesos
La red mejora progresivamente en cada ejecucion
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import json
from src.mlp import MLP
from src.dataset import DatasetGenerator
from src.complexity_analyzer import ComplexityAnalyzer


# Configuracion global
WEIGHTS_CHECKPOINT_PATH = 'model_state/weights_checkpoint.json'
TRAINING_HISTORY_PATH = 'model_state/training_history.json'


def load_training_history():
    """Carga el historico de entrenamiento previo"""
    if os.path.exists(TRAINING_HISTORY_PATH):
        try:
            with open(TRAINING_HISTORY_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        'total_executions': 0,
        'total_epochs_trained': 0,
        'all_losses': [],
        'all_accuracies': [],
        'execution_history': []
    }


def save_training_history(history):
    """Guarda el historico de entrenamiento"""
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    with open(TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100 + "\n")


def print_subheader(title):
    """Imprime un subencabezado"""
    print("\n" + "-"*100)
    print(f"  {title}")
    print("-"*100 + "\n")


def print_progress_bar(epoch, total_epochs, loss, accuracy, elapsed, eta):
    """Imprime una barra de progreso visual"""
    progress = epoch / total_epochs
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '#' * filled + '.' * (bar_length - filled)
    
    percentage = progress * 100
    
    print(f"[{bar}] {percentage:5.1f}% | "
          f"Epoca {epoch:4d}/{total_epochs} | "
          f"Perdida: {loss:.6f} | "
          f"Precision: {accuracy:.4f} | "
          f"Tiempo: {elapsed:6.1f}s | "
          f"ETA: {eta:6.1f}s")


def train_with_detailed_progress(model, X_train, y_train, X_val, y_val,
                                epochs=5, batch_size=32,
                                learning_rate=0.01, 
                                experiment_title="ENTRENAMIENTO",
                                optimizer="SGD"):
    """Entrena el modelo mostrando el progreso detallado de cada epoca"""
    
    print(f"\n{'='*100}")
    print(f"[ENTRENAMIENTO {experiment_title}]")
    print(f"Continuacion del entrenamiento con aprendizaje progresivo")
    print(f"{'='*100}")
    
    # Encabezado 1: INFORMACION DEL EXPERIMENTO
    print_subheader("1. ENCABEZADO DEL ENTRENAMIENTO")
    print(f"  * Titulo del Experimento: {experiment_title}")
    print(f"  * Arquitectura de la Red: Input({X_train.cols}) -> Hidden(64) -> Hidden(64) -> Output({y_train.cols})")
    print(f"  * Tamano del Dataset: {X_train.rows} muestras de entrenamiento + {X_val.rows} de validacion")
    print(f"  * Metodo de Optimizacion: {optimizer}")
    
    print(f"\nConfiguracion:")
    print(f"  - Tamano de lote (batch size): {batch_size}")
    print(f"  - Tasa de aprendizaje (learning rate): {learning_rate}")
    print(f"  - Muestras de entrenamiento: {X_train.rows}")
    print(f"  - Muestras de validacion: {X_val.rows}")
    print(f"  - Total de parametros: {model.get_total_params():,}")
    print(f"\nIniciando entrenamiento...\n")
    
    results = {
        'losses': [],
        'accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'epoch_times': [],
        'total_time': 0,
        'epochs_in_this_run': epochs
    }
    
    start_total = time.time()
    times_for_eta = []
    
    # Seccion 3: INFORMACION POR CADA EPOCA
    print_subheader("3. INFORMACION POR CADA EPOCA")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Entrenar epoca
        train_loss, train_acc = model.train_epoch(X_train, y_train, batch_size, learning_rate)
        
        epoch_time = time.time() - epoch_start
        times_for_eta.append(epoch_time)
        
        # Mantener promedio de ultimos 10 tiempos para ETA mas preciso
        if len(times_for_eta) > 10:
            times_for_eta = times_for_eta[-10:]
        
        avg_epoch_time = sum(times_for_eta) / len(times_for_eta)
        remaining_epochs = epochs - epoch - 1
        eta = avg_epoch_time * remaining_epochs
        
        results['losses'].append(train_loss)
        results['accuracies'].append(train_acc)
        results['epoch_times'].append(epoch_time)
        
        # Validacion
        val_predictions = model.forward(X_val)
        val_loss = model.compute_loss(val_predictions, y_val)
        val_acc = model.compute_accuracy(val_predictions, y_val)
        
        results['val_losses'].append(val_loss)
        results['val_accuracies'].append(val_acc)
        
        # Mostrar progreso
        total_elapsed = time.time() - start_total
        print_progress_bar(epoch + 1, epochs, train_loss, train_acc, total_elapsed, eta)
        
        # Mostrar informacion detallada cada 100 epocas (o siempre si son menos de 100)
        show_detail = False
        if epochs < 100:
            show_detail = (epoch == 0) or (epoch == epochs - 1)
        else:
            show_detail = ((epoch + 1) % 100 == 0) or (epoch == 0) or (epoch == epochs - 1)
        
        if show_detail:
            print()
            print(f"\n  L- Epoca {epoch+1:4d}: "
                  f"Perdida={train_loss:.6f} | "
                  f"Precision={train_acc:.4f} | "
                  f"Val Perdida={val_loss:.6f} | "
                  f"Val Precision={val_acc:.4f} | "
                  f"Tiempo={epoch_time:.3f}s")
    
    results['total_time'] = time.time() - start_total
    
    print("\n\n" + "="*100)
    print("  EPOCA ACTUAL COMPLETADA")
    print("="*100)
    
    return results


def print_training_summary(results, initial_loss, final_loss,
                          total_epochs_ever, complexity, 
                          training_history):
    """Imprime un resumen completo del entrenamiento con la lista requerida"""
    
    print_subheader("4. RESULTADOS FINALES DEL ENTRENAMIENTO")
    
    print(f"  * Numero total de epocas ENTRENADAS HASTA AHORA: {total_epochs_ever}")
    print(f"  * Epocas en esta ejecucion: {results['epochs_in_this_run']}")
    print(f"  * Perdida final obtenida en esta ejecucion: {final_loss:.6f}")
    print(f"  * Exactitud final obtenida en esta ejecucion: {results['accuracies'][-1]:.4f}")
    print(f"  * Tiempo total en esta ejecucion: {results['total_time']:.2f} segundos")
    print(f"  * Tiempo promedio por epoca: {sum(results['epoch_times']) / len(results['epoch_times']):.4f}s")
    
    # Informacion adicional de progresion
    print(f"\n  Progresion General:")
    if training_history['execution_history']:
        print(f"  * Perdida Inicial (primera ejecucion): {training_history['execution_history'][0]['initial_loss']:.6f}")
    print(f"  * Perdida en esta ejecucion: {final_loss:.6f}")
    if training_history['all_accuracies']:
        print(f"  * Mejor Precision registrada: {max(training_history['all_accuracies']):.4f}")
    
    max_acc = max(results['accuracies'])
    max_acc_epoch = results['accuracies'].index(max_acc) + 1
    
    final_acc = results['accuracies'][-1]
    
    print(f"\n  Metricas de entrenamiento en esta ejecucion:")
    print(f"    - Exactitud Inicial: {results['accuracies'][0]:.4f}")
    print(f"    - Exactitud Maxima: {max_acc:.4f} (epoca {max_acc_epoch})")
    print(f"    - Exactitud Final: {final_acc:.4f}")
    
    max_val_acc = max(results['val_accuracies'])
    max_val_acc_epoch = results['val_accuracies'].index(max_val_acc) + 1
    final_val_acc = results['val_accuracies'][-1]
    
    print(f"\n  Metricas de validacion en esta ejecucion:")
    print(f"    - Exactitud Inicial: {results['val_accuracies'][0]:.4f}")
    print(f"    - Exactitud Maxima: {max_val_acc:.4f} (epoca {max_val_acc_epoch})")
    print(f"    - Exactitud Final: {final_val_acc:.4f}")
    
    total_time = results['total_time']
    avg_epoch_time = sum(results['epoch_times']) / len(results['epoch_times'])
    min_epoch_time = min(results['epoch_times'])
    max_epoch_time = max(results['epoch_times'])
    
    print(f"\n  Tiempos:")
    print(f"    - Total esta ejecucion: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"    - Por epoca (promedio): {avg_epoch_time:.4f}s")
    print(f"    - Por epoca (rango): {min_epoch_time:.4f}s - {max_epoch_time:.4f}s")


def analyze_convergence(results):
    """Analiza la convergencia del modelo"""
    
    print_subheader("ANALISIS DE CONVERGENCIA")
    
    losses = results['losses']
    val_losses = results['val_losses']
    
    # Calcular tasa de cambio promedio
    loss_changes = []
    for i in range(1, min(10, len(losses))):
        change = abs(losses[i] - losses[i-1])
        loss_changes.append(change)
    
    if loss_changes:
        avg_initial_change = sum(loss_changes) / len(loss_changes)
        print(f"  * Cambio promedio de perdida en primeras epocas: {avg_initial_change:.8f}")
    
    # Cambio en las ultimas epocas
    loss_changes_final = []
    for i in range(max(len(losses)-10, 1), len(losses)):
        change = abs(losses[i] - losses[i-1])
        loss_changes_final.append(change)
    
    if loss_changes_final:
        avg_final_change = sum(loss_changes_final) / len(loss_changes_final)
        print(f"  * Cambio promedio de perdida en ultimas epocas: {avg_final_change:.8f}")
    
    # Detectar sobreajuste (overfitting)
    if len(results['val_losses']) > 5:
        train_loss_final = sum(losses[-5:]) / 5
        val_loss_final = sum(val_losses[-5:]) / 5
        
        overfitting_ratio = val_loss_final / train_loss_final if train_loss_final > 0 else 1.0
        
        print(f"\n  Perdida promedio ultimas 5 epocas:")
        print(f"    - Entrenamiento: {train_loss_final:.6f}")
        print(f"    - Validacion: {val_loss_final:.6f}")
        print(f"    - Ratio (Val/Train): {overfitting_ratio:.4f}")
        
        if overfitting_ratio > 1.1:
            print(f"    [!] Posible sobreajuste detectado")
        else:
            print(f"    [OK] Modelo generaliza bien")


def main():
    """Funcion principal"""
    
    print_header("RED NEURONAL - ANALISIS DE COMPLEJIDAD Y ENTRENAMIENTO CONTINUO")
    
    # Configuracion
    EPOCHS_PER_RUN = 100  # 100 epocas para mejor entrenamiento
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    HIDDEN_SIZE = 64
    NUM_SAMPLES = 1000
    NUM_FEATURES = 28 * 28  # MNIST-like
    NUM_CLASSES = 10
    
    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]
    
    # Cargar historico de entrenamiento previo
    training_history = load_training_history()
    model_exists = os.path.exists(WEIGHTS_CHECKPOINT_PATH)
    
    # MOSTRAR ANALISIS TEORICO DE COMPLEJIDAD UNA SOLA VEZ (al inicio)
    if training_history['total_executions'] == 0:
        print_subheader("2. ANALISIS ALGORITMICO (UNA SOLA VEZ AL INICIO)")
        
        complexity = ComplexityAnalyzer.theoretical_complexity_analysis(
            layer_sizes=layer_sizes,
            batch_size=BATCH_SIZE,
            num_epochs=EPOCHS_PER_RUN,
            num_training_samples=NUM_SAMPLES
        )
        
        print(f"Arquitectura de la Red:")
        print(f"  * Capas: {' -> '.join(map(str, layer_sizes))}")
        print(f"  * Total de parametros: {complexity['total_parameters']:,}")
        
        print(f"\nComplejidad Temporal:")
        print(f"  * Forward pass por lote: O(B x suma(I_i x O_i)) = O({complexity['forward_cost_per_batch']:,})")
        print(f"  * Backward pass por lote: O({complexity['backward_cost_per_batch']:,})")
        print(f"  * Costo por epoca: O({complexity['cost_per_epoch']:,})")
        print(f"  * Notacion: {complexity['time_complexity_notation']}")
        print(f"  * Operaciones totales estimadas: ~{complexity['total_time_operations']:,} ops")
        
        print(f"\nComplejidad Espacial:")
        print(f"  * Parametros + gradientes: O({complexity['params_space']:,})")
        print(f"  * Activaciones en memoria: O({complexity['activation_space']:,})")
        print(f"  * Cache para backprop: O({complexity['cache_space']:,})")
        print(f"  * Notacion: {complexity['space_complexity_notation']}")
        print(f"  * Total: ~{complexity['total_space_complexity']:,} floats")
        
        # Guardar complejidad para futuras ejecuciones
        os.makedirs('model_state', exist_ok=True)
        with open('model_state/complexity_analysis.json', 'w') as f:
            json.dump(complexity, f, indent=2)
    else:
        # Cargar complejidad anterior
        try:
            with open('model_state/complexity_analysis.json', 'r') as f:
                complexity = json.load(f)
        except:
            complexity = {}
        print(f"\nSe reutiliza analisis de complejidad previo")
        print(f"  * Ejecuciones anteriores: {training_history['total_executions']}")
        print(f"  * Epocas totales entrenadas hasta ahora: {training_history['total_epochs_trained']}")
    
    # Preparar dataset
    print_header("PREPARACION DEL DATASET")
    
    print(f"Generando dataset...")
    X, y = DatasetGenerator.generate_mnist_like(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        img_size=28,
        seed=42
    )
    
    print(f"Normalizando datos...")
    X = DatasetGenerator.normalize_dataset(X)
    
    print(f"Dividiendo en entrenamiento y validacion...")
    X_train, y_train, X_val, y_val = DatasetGenerator.split_dataset(
        X, y, train_ratio=0.8, seed=42
    )
    
    print(f"\nDataset listo:")
    print(f"  - Muestras de entrenamiento: {X_train.rows}")
    print(f"  - Muestras de validacion: {X_val.rows}")
    print(f"  - Caracteristicas: {X_train.cols}")
    print(f"  - Clases: {y_train.cols}")
    
    # Crear modelo
    print_header("CREACION/CARGA DEL MODELO")
    
    model = MLP(
        layer_sizes=layer_sizes,
        activations=['relu', 'relu', 'linear'],
        loss_function='cross_entropy'
    )
    
    print(f"Modelo creado:")
    print(f"  - Capas: {len(model.layers)}")
    print(f"  - Parametros totales: {model.get_total_params():,}")
    
    # Cargar pesos previos si existen
    if model_exists:
        if model.load_weights(WEIGHTS_CHECKPOINT_PATH):
            print(f"\nPesos cargados desde ejecucion anterior")
            print(f"  La red neuronal continua aprendiendo con los pesos previos")
        else:
            print(f"\n[!] No se pudieron cargar pesos previos, iniciando desde cero")
    else:
        print(f"\nPrimera ejecucion: inicializando pesos aleatorios")
    
    # Entrenar
    print_header("ENTRENAMIENTO")
    
    initial_loss = None
    results = train_with_detailed_progress(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS_PER_RUN,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        experiment_title=f"#{training_history['total_executions'] + 1}",
        optimizer="Stochastic Gradient Descent (SGD)"
    )
    
    initial_loss = results['losses'][0]
    final_loss = results['losses'][-1]
    
    # Resumen y analisis
    total_epochs_ever = training_history['total_epochs_trained'] + EPOCHS_PER_RUN
    
    print_training_summary(
        results, 
        initial_loss, 
        final_loss,
        total_epochs_ever,
        complexity,
        training_history
    )
    
    analyze_convergence(results)
    
    # Evaluacion final
    print_header("EVALUACION FINAL")
    
    final_predictions = model.forward(X_val)
    final_accuracy = model.compute_accuracy(final_predictions, y_val)
    
    print(f"\nPrecision Final en Validacion: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Guardar pesos para proxima ejecucion
    print_header("PERSISTENCIA DEL MODELO")
    
    if model.save_weights(WEIGHTS_CHECKPOINT_PATH):
        print(f"Pesos guardados para proxima ejecucion")
        print(f"  En la proxima ejecucion, el modelo continuara desde estos pesos")
    else:
        print(f"[!] Error al guardar pesos")
    
    # Actualizar historico de entrenamiento
    execution_record = {
        'execution_number': training_history['total_executions'] + 1,
        'epochs_in_run': EPOCHS_PER_RUN,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'val_final_accuracy': results['val_accuracies'][-1],
        'total_time': results['total_time'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    training_history['total_executions'] += 1
    training_history['total_epochs_trained'] = total_epochs_ever
    training_history['execution_history'].append(execution_record)
    training_history['all_losses'].extend(results['losses'])
    training_history['all_accuracies'].extend(results['accuracies'])
    
    save_training_history(training_history)
    
    print_header("RESUMEN DE PROGRESION")
    
    print(f"\n  Ejecuciones completadas: {training_history['total_executions']}")
    print(f"  Epocas totales entrenadas: {training_history['total_epochs_trained']}")
    
    if len(training_history['execution_history']) > 1:
        prev_exec = training_history['execution_history'][-2]
        curr_exec = training_history['execution_history'][-1]
        
        loss_improvement = prev_exec['final_loss'] - curr_exec['final_loss']
        acc_improvement = curr_exec['final_accuracy'] - prev_exec['final_accuracy']
        
        print(f"\n  Comparacion con ejecucion anterior:")
        print(f"    * Cambio en perdida: {loss_improvement:+.6f}")
        print(f"    * Cambio en precision: {acc_improvement:+.4f}")
        
        if loss_improvement > 0:
            print(f"    La perdida mejoro (disminuyo)")
        elif loss_improvement < 0:
            print(f"    [!] La perdida empeoro (aumento)")
        
        if acc_improvement > 0:
            print(f"    La precision mejoro")
        elif acc_improvement < 0:
            print(f"    [!] La precision empeoro")
    
    print(f"\n  Para continuar el aprendizaje, ejecuta el programa nuevamente.")
    print(f"  Cada ejecucion agregara {EPOCHS_PER_RUN} epocas adicionales de entrenamiento.")
    
    print("\n" + "="*100)
    print("  ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("  La red neuronal ha mejorado y sus pesos fueron guardados para la proxima ejecucion")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
