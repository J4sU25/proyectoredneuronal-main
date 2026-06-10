"""
Experimentos de Escalado - Analisis empirico de complejidad
Varia B (batch_size), h (hidden_size), E (epochs) y N (muestras)

Fase 1 (RA1): Entrega de experimentos de escalado
"""

import time
import math
from typing import List, Dict, Tuple


# ============================================================
# EXPERIMENTO 1: Tiempo vs N (numero de muestras)
# Expectativa teorica: tiempo ~ O(N) para N fijo B, h, E
# ============================================================

def experimento_escalar_N(layer_sizes_base: List[int],
                          N_values: List[int],
                          batch_size: int = 32,
                          epochs: int = 5,
                          learning_rate: float = 0.01) -> Dict:
    """
    Mide como escala el tiempo de entrenamiento con N (muestras).

    Expectativa teorica:
    - Costo por epoca = ceil(N/B) * costo_por_batch
    - Costo total ~ O(E * N/B * sum(I_i * O_i))
    - Para E, B, h fijos: Tiempo ~ O(N)

    Recurrencia: No aplica directamente (es iterativo)
    Relacion lineal esperada: t(N) = c * N
    """
    from src.dataset import DatasetGenerator
    from src.mlp import MLP

    results = {
        'N_values': N_values,
        'times': [],
        'samples_per_sec': [],
        'losses': []
    }

    for N in N_values:
        X, y = DatasetGenerator.generate_synthetic_classification(
            num_samples=N,
            num_features=layer_sizes_base[0],
            num_classes=layer_sizes_base[-1],
            seed=42
        )

        model = MLP(layer_sizes_base,
                    ['relu'] * (len(layer_sizes_base) - 2) + ['linear'],
                    loss_function='cross_entropy')

        inicio = time.perf_counter()
        for _ in range(epochs):
            loss, _ = model.train_epoch(X, y, batch_size, learning_rate)
        fin = time.perf_counter()

        elapsed = fin - inicio
        sps = (N * epochs) / elapsed if elapsed > 0 else 0

        results['times'].append(elapsed)
        results['samples_per_sec'].append(sps)
        results['losses'].append(loss)

    return results


# ============================================================
# EXPERIMENTO 2: Tiempo vs B (batch_size)
# Expectativa teorica: tiempo ~ O(1/B) para mismo N, E, h
# Mas batches con B pequeno = mas actualizaciones pero mas overhead
# ============================================================

def experimento_escalar_B(layer_sizes_base: List[int],
                          N: int,
                          B_values: List[int],
                          epochs: int = 5,
                          learning_rate: float = 0.01) -> Dict:
    """
    Mide como cambia el tiempo con B (batch_size).

    Expectativa teorica:
    - Numero de batches por epoca = ceil(N/B)
    - Tiempo por epoch ~ ceil(N/B) * O(B * sum(I_i * O_i))
    - Simplificando: ~ O(N * sum(I_i * O_i)) independiente de B
    - En la practica: B grande = mejor uso de cache, menos overhead

    No hay recurrencia asociada directamente.
    """
    from src.dataset import DatasetGenerator
    from src.mlp import MLP

    # Generar datos una sola vez
    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=N,
        num_features=layer_sizes_base[0],
        num_classes=layer_sizes_base[-1],
        seed=42
    )

    results = {
        'B_values': B_values,
        'times': [],
        'throughput': [],
        'num_batches': []
    }

    for B in B_values:
        model = MLP(layer_sizes_base,
                    ['relu'] * (len(layer_sizes_base) - 2) + ['linear'],
                    loss_function='cross_entropy')

        num_batches = math.ceil(N / B)

        inicio = time.perf_counter()
        for _ in range(epochs):
            model.train_epoch(X, y, B, learning_rate)
        fin = time.perf_counter()

        elapsed = fin - inicio
        throughput = (N * epochs) / elapsed if elapsed > 0 else 0

        results['times'].append(elapsed)
        results['throughput'].append(throughput)
        results['num_batches'].append(num_batches)

    return results


# ============================================================
# EXPERIMENTO 3: Tiempo vs h (hidden_size)
# Expectativa teorica: tiempo ~ O(h^2) aproximadamente
# ya que aparece en dos terminos: I*h + h*O
# ============================================================

def experimento_escalar_h(input_size: int,
                          output_size: int,
                          N: int,
                          h_values: List[int],
                          batch_size: int = 32,
                          epochs: int = 5,
                          learning_rate: float = 0.01) -> Dict:
    """
    Mide como escala el tiempo con h (tamano de capa oculta).

    Para una red [I, h, O]:
    - Parametros = I*h + h + h*O + O = h*(I+O) + h + O
    - Costo por batch = O(B*I*h + B*h*O) = O(B*h*(I+O))
    - Para I, O, B, N, E fijos: Tiempo ~ O(h)

    Para dos capas ocultas [I, h, h, O]:
    - Costo por batch = O(B*I*h + B*h*h + B*h*O) = O(B*h*(I+h+O))
    - Tiempo ~ O(h^2) por el termino B*h*h
    """
    from src.dataset import DatasetGenerator
    from src.mlp import MLP

    # Generar datos una sola vez
    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=N,
        num_features=input_size,
        num_classes=output_size,
        seed=42
    )

    results = {
        'h_values': h_values,
        'times': [],
        'params': [],
        'time_per_param': []
    }

    for h in h_values:
        layer_sizes = [input_size, h, h, output_size]
        model = MLP(layer_sizes, ['relu', 'relu', 'linear'],
                    loss_function='cross_entropy')
        num_params = model.get_total_params()

        inicio = time.perf_counter()
        for _ in range(epochs):
            model.train_epoch(X, y, batch_size, learning_rate)
        fin = time.perf_counter()

        elapsed = fin - inicio
        tpp = elapsed / num_params if num_params > 0 else 0

        results['times'].append(elapsed)
        results['params'].append(num_params)
        results['time_per_param'].append(tpp)

    return results


# ============================================================
# EXPERIMENTO 4: Tiempo vs E (epochs)
# Expectativa teorica: tiempo ~ O(E) - relacion perfectamente lineal
# ============================================================

def experimento_escalar_E(layer_sizes_base: List[int],
                          N: int,
                          E_values: List[int],
                          batch_size: int = 32,
                          learning_rate: float = 0.01) -> Dict:
    """
    Mide como escala el tiempo con E (epocas).

    Expectativa teorica: Tiempo = E * costo_por_epoca = O(E)
    Relacion perfectamente lineal: t(E) = c * E

    Es la relacion mas facil de verificar empiricamente.
    """
    from src.dataset import DatasetGenerator
    from src.mlp import MLP

    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=N,
        num_features=layer_sizes_base[0],
        num_classes=layer_sizes_base[-1],
        seed=42
    )

    model = MLP(layer_sizes_base,
                ['relu'] * (len(layer_sizes_base) - 2) + ['linear'],
                loss_function='cross_entropy')

    results = {
        'E_values': E_values,
        'times': [],
        'time_per_epoch': []
    }

    for E in E_values:
        # Reiniciar modelo para comparacion justa
        from src.mlp import MLP
        model_e = MLP(layer_sizes_base,
                      ['relu'] * (len(layer_sizes_base) - 2) + ['linear'],
                      loss_function='cross_entropy')

        inicio = time.perf_counter()
        for _ in range(E):
            model_e.train_epoch(X, y, batch_size, learning_rate)
        fin = time.perf_counter()

        elapsed = fin - inicio
        tpe = elapsed / E if E > 0 else 0

        results['times'].append(elapsed)
        results['time_per_epoch'].append(tpe)

    return results


# ============================================================
# IMPRESION DE RESULTADOS CON GRAFICAS ASCII
# ============================================================

def _barra_ascii(valor: float, max_valor: float, ancho: int = 40) -> str:
    """Genera una barra ASCII proporcional al valor."""
    if max_valor <= 0:
        return "." * ancho
    n_llenos = int(ancho * valor / max_valor)
    return "#" * n_llenos + "." * (ancho - n_llenos)


def imprimir_experimento_N(results: Dict) -> None:
    """Imprime resultados del experimento de escalado por N."""
    print("\n" + "=" * 80)
    print("  EXPERIMENTO RA1: Tiempo vs N (numero de muestras)")
    print("  Expectativa teorica: Tiempo ~ O(N) (lineal)")
    print("=" * 80)

    N_values = results['N_values']
    times = results['times']
    max_t = max(times) if times else 1

    print(f"\n  {'N':>6} | {'Tiempo (s)':>10} | {'Muestras/s':>12} | Grafica (proporcional)")
    print("  " + "-" * 72)

    for i, N in enumerate(N_values):
        t = times[i]
        sps = results['samples_per_sec'][i]
        barra = _barra_ascii(t, max_t)
        print(f"  {N:>6} | {t:>10.4f} | {sps:>12.1f} | {barra}")

    # Analisis de linearidad
    if len(N_values) >= 2:
        ratio_N = N_values[-1] / N_values[0]
        ratio_t = times[-1] / times[0] if times[0] > 0 else 0
        print(f"\n  Analisis de linealidad:")
        print(f"    N escala {ratio_N:.1f}x, tiempo escala {ratio_t:.1f}x")
        if abs(ratio_t / ratio_N - 1) < 0.3:
            print(f"    [OK] Comportamiento lineal confirmado (ratio ~ {ratio_t/ratio_N:.2f})")
        else:
            print(f"    [!] Comportamiento sub/super-lineal (ratio = {ratio_t/ratio_N:.2f})")


def imprimir_experimento_B(results: Dict) -> None:
    """Imprime resultados del experimento de escalado por B."""
    print("\n" + "=" * 80)
    print("  EXPERIMENTO RA1: Tiempo vs B (batch_size)")
    print("  Expectativa: Tiempo aproximadamente constante (variacion por overhead)")
    print("=" * 80)

    B_values = results['B_values']
    times = results['times']
    max_t = max(times) if times else 1

    print(f"\n  {'B':>6} | {'Tiempo (s)':>10} | {'Throughput':>12} | {'#Batches':>8} | Grafica")
    print("  " + "-" * 80)

    for i, B in enumerate(B_values):
        t = times[i]
        tp = results['throughput'][i]
        nb = results['num_batches'][i]
        barra = _barra_ascii(t, max_t)
        print(f"  {B:>6} | {t:>10.4f} | {tp:>12.1f} | {nb:>8} | {barra}")


def imprimir_experimento_h(results: Dict) -> None:
    """Imprime resultados del experimento de escalado por h."""
    print("\n" + "=" * 80)
    print("  EXPERIMENTO RA1: Tiempo vs h (hidden_size)")
    print("  Expectativa teorica: Tiempo ~ O(h^2) con dos capas ocultas")
    print("=" * 80)

    h_values = results['h_values']
    times = results['times']
    params = results['params']
    max_t = max(times) if times else 1

    print(f"\n  {'h':>6} | {'Tiempo (s)':>10} | {'Parametros':>12} | Grafica")
    print("  " + "-" * 70)

    for i, h in enumerate(h_values):
        t = times[i]
        p = params[i]
        barra = _barra_ascii(t, max_t)
        print(f"  {h:>6} | {t:>10.4f} | {p:>12,} | {barra}")

    # Analisis de cuadraticidad
    if len(h_values) >= 2:
        ratio_h = h_values[-1] / h_values[0]
        ratio_t = times[-1] / times[0] if times[0] > 0 else 0
        print(f"\n  Analisis de cuadraticidad:")
        print(f"    h escala {ratio_h:.1f}x, tiempo escala {ratio_t:.1f}x")
        print(f"    h^2 prediccion: {ratio_h**2:.1f}x | Medido: {ratio_t:.1f}x")
        print(f"    h^1 prediccion: {ratio_h:.1f}x  | (terminos lineales dominantes para h pequeno)")


def imprimir_experimento_E(results: Dict) -> None:
    """Imprime resultados del experimento de escalado por E."""
    print("\n" + "=" * 80)
    print("  EXPERIMENTO RA1: Tiempo vs E (epocas)")
    print("  Expectativa teorica: Tiempo ~ O(E) (relacion perfectamente lineal)")
    print("=" * 80)

    E_values = results['E_values']
    times = results['times']
    tpe = results['time_per_epoch']
    max_t = max(times) if times else 1

    print(f"\n  {'E':>6} | {'Tiempo total':>12} | {'Tiempo/epoca':>13} | Grafica")
    print("  " + "-" * 70)

    for i, E in enumerate(E_values):
        t = times[i]
        tp = tpe[i]
        barra = _barra_ascii(t, max_t)
        print(f"  {E:>6} | {t:>10.4f} s | {tp:>11.4f} s | {barra}")

    # Verificar linealidad
    if len(E_values) >= 2:
        ratio_E = E_values[-1] / E_values[0]
        ratio_t = times[-1] / times[0] if times[0] > 0 else 0
        print(f"\n  Analisis de linealidad:")
        print(f"    E escala {ratio_E:.1f}x, tiempo escala {ratio_t:.1f}x")
        print(f"    Tiempo/epoca (casi constante): "
              f"{min(tpe):.4f}s a {max(tpe):.4f}s")
        if abs(ratio_t / ratio_E - 1) < 0.2:
            print(f"    [OK] Relacion lineal con E confirmada")
        else:
            print(f"    [!] Variacion inesperada (ratio = {ratio_t/ratio_E:.2f})")


def ejecutar_todos_experimentos(layer_sizes: List[int],
                                 N_base: int = 500) -> Dict:
    """
    Ejecuta los cuatro experimentos de escalado (RA1).
    Usa valores moderados para no tardar demasiado.
    """
    input_size = layer_sizes[0]
    output_size = layer_sizes[-1]

    print("\n" + "=" * 80)
    print("  RA1: EXPERIMENTOS DE ESCALADO (B, h, E, N)")
    print("  Valida empiricamente las derivaciones teoricas de complejidad")
    print("=" * 80)

    results_all = {}

    # -- Experimento N --
    print("\n  [1/4] Ejecutando experimento: Tiempo vs N...")
    N_values = [100, 200, 400, 800, 1000]
    r_N = experimento_escalar_N(layer_sizes, N_values, batch_size=32, epochs=3)
    imprimir_experimento_N(r_N)
    results_all['N'] = r_N

    # -- Experimento B --
    print("\n  [2/4] Ejecutando experimento: Tiempo vs B...")
    B_values = [8, 16, 32, 64, 128]
    r_B = experimento_escalar_B(layer_sizes, N=N_base,
                                 B_values=B_values, epochs=3)
    imprimir_experimento_B(r_B)
    results_all['B'] = r_B

    # -- Experimento h --
    print("\n  [3/4] Ejecutando experimento: Tiempo vs h...")
    h_values = [8, 16, 32, 64, 128]
    r_h = experimento_escalar_h(input_size, output_size, N=N_base,
                                 h_values=h_values, epochs=3)
    imprimir_experimento_h(r_h)
    results_all['h'] = r_h

    # -- Experimento E --
    print("\n  [4/4] Ejecutando experimento: Tiempo vs E...")
    E_values = [2, 5, 10, 20, 30]
    r_E = experimento_escalar_E(layer_sizes, N=N_base,
                                 E_values=E_values, batch_size=32)
    imprimir_experimento_E(r_E)
    results_all['E'] = r_E

    return results_all
