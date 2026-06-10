"""
Algoritmos de Ordenamiento - Implementacion desde cero
Heapsort y Quicksort con analisis de complejidad y comparativa empirica

Fase 4: Baseline y comparacion de algoritmos auxiliares
"""

import time
import random
from typing import List, Tuple


# ============================================================
# HEAPSORT - O(n log n) garantizado en todos los casos
# Recurrencia: T(n) = T(n/2) + O(n)  -> O(n log n)
# ============================================================

def _heapify(arr: List[float], n: int, i: int) -> None:
    """
    Mantiene la propiedad de max-heap desde el indice i.

    Recurrencia: T(n) = T(n/2) + O(1)
    Aplicando Metodo Maestro: a=1, b=2, f(n)=O(1)
    n^(log_b(a)) = n^0 = 1 = O(1)  -> Caso 2: T(n) = O(log n)

    Complejidad temporal: O(log n)
    Complejidad espacial: O(log n) por recursion en pila
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


def heapsort(arr: List[float]) -> List[float]:
    """
    Heapsort - Ordenamiento basado en Max-Heap.

    Analisis de complejidad:
    - Fase 1 (build_heap): O(n) - Metodo de Floyd
        Se aplica heapify a cada nodo desde n//2-1 hasta 0
        Suma geometrica: sum_{h=0}^{log n} (n / 2^h) * O(h) = O(n)
    - Fase 2 (extracciones): n extracciones de O(log n) c/u = O(n log n)
    - Total: O(n) + O(n log n) = O(n log n)

    Recurrencia del heapify:
    T(n) = T(n/2) + O(1)  => Metodo Maestro Caso 2 => T(n) = O(log n)

    Big-O:    O(n log n) - En todos los casos (mejor, promedio, peor)
    Big-Omega: Omega(n log n) - Cota inferior ajustada
    Big-Theta: Theta(n log n) - Cota exacta

    Complejidad espacial: O(log n) por recursion en pila de heapify
    (In-place: no requiere memoria adicional proporcional a n)
    """
    arr = arr[:]  # copia para no modificar original
    n = len(arr)

    # Fase 1: Construir max-heap - O(n)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Fase 2: Extraer elementos del heap - O(n log n)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]   # Mover raiz al final
        _heapify(arr, i, 0)                # Restaurar heap

    return arr


# ============================================================
# QUICKSORT - O(n log n) promedio, O(n^2) peor caso
# Recurrencia promedio: T(n) = 2T(n/2) + O(n) -> O(n log n)
# ============================================================

def _partition(arr: List[float], low: int, high: int) -> int:
    """
    Particiona el arreglo usando el ultimo elemento como pivote.

    Recurrencia: O(n) por particion (recorre todos los elementos)

    Complejidad temporal: O(n) donde n = high - low + 1
    Complejidad espacial: O(1)
    """
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def _quicksort_recursive(arr: List[float], low: int, high: int) -> None:
    """
    Implementacion recursiva de Quicksort.

    Recurrencia:
    - Caso promedio (pivote divide en mitades): T(n) = 2T(n/2) + O(n)
      Metodo Maestro: a=2, b=2, f(n)=O(n)
      n^(log_b(a)) = n^1 = n = Theta(n)  -> Caso 2: T(n) = O(n log n)

    - Caso peor (arreglo ya ordenado, pivote en extremo): T(n) = T(n-1) + O(n)
      Recurrencia lineal: T(n) = O(n^2)

    - Caso mejor (pivote divide exactamente en mitades): T(n) = O(n log n)

    Complejidad temporal: O(n log n) promedio, O(n^2) peor caso
    Complejidad espacial: O(log n) promedio en pila, O(n) peor caso
    """
    if low < high:
        pivot_idx = _partition(arr, low, high)
        _quicksort_recursive(arr, low, pivot_idx - 1)
        _quicksort_recursive(arr, pivot_idx + 1, high)


def quicksort(arr: List[float]) -> List[float]:
    """
    Quicksort con pivote en ultimo elemento.

    Big-O:    O(n^2) - Peor caso (arreglo ya ordenado)
    Big-Omega: Omega(n log n) - Mejor caso (pivote siempre en medio)
    Big-Theta: Theta(n log n) - Caso promedio con pivote aleatorio

    Nota: Usar pivote aleatorio convierte el peor caso en esperanza O(n log n)
    """
    arr = arr[:]
    _quicksort_recursive(arr, 0, len(arr) - 1)
    return arr


def quicksort_random_pivot(arr: List[float]) -> List[float]:
    """
    Quicksort con pivote aleatorio para evitar O(n^2) en arreglos ordenados.

    Con pivote aleatorio: T(n) = O(n log n) con alta probabilidad
    La esperanza del numero de comparaciones es 2n*ln(n) ~ 1.39 n log n
    """
    def _partition_random(a: List[float], low: int, high: int) -> int:
        # Elegir pivote aleatorio y moverlo al final
        rand_idx = random.randint(low, high)
        a[rand_idx], a[high] = a[high], a[rand_idx]
        return _partition(a, low, high)

    def _qsort(a: List[float], low: int, high: int) -> None:
        if low < high:
            pi = _partition_random(a, low, high)
            _qsort(a, low, pi - 1)
            _qsort(a, pi + 1, high)

    arr = arr[:]
    _qsort(arr, 0, len(arr) - 1)
    return arr


# ============================================================
# COMPARATIVA EMPIRICA: Heapsort vs Quicksort
# ============================================================

def comparar_heapsort_vs_quicksort(sizes: List[int] = None,
                                   repetitions: int = 3) -> dict:
    """
    Compara empiricamente Heapsort vs Quicksort en diferentes tamanios.

    Genera tres tipos de datos:
    - Aleatorio: caso promedio para ambos
    - Ordenado: peor caso para Quicksort basico, normal para Heapsort
    - Inverso: otro peor caso para Quicksort basico

    Complejidad de este analisis: O(k * n * log n * repetitions)
    donde k = numero de tamanios probados

    Retorna: diccionario con tiempos medidos por algoritmo y tipo de dato
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]

    results = {
        'sizes': sizes,
        'heapsort_random': [],
        'quicksort_random': [],
        'quicksort_rand_pivot_random': [],
        'heapsort_sorted': [],
        'quicksort_rand_pivot_sorted': [],
    }

    for n in sizes:
        # --- Datos aleatorios ---
        data_random = [random.uniform(0, 1000) for _ in range(n)]

        t_heap = _medir_tiempo(heapsort, data_random, repetitions)
        t_quick = _medir_tiempo(quicksort, data_random, repetitions)
        t_quick_rand = _medir_tiempo(quicksort_random_pivot, data_random, repetitions)

        results['heapsort_random'].append(t_heap)
        results['quicksort_random'].append(t_quick)
        results['quicksort_rand_pivot_random'].append(t_quick_rand)

        # --- Datos ordenados (peor caso para quicksort sin pivot random) ---
        data_sorted = sorted(data_random)

        t_heap_s = _medir_tiempo(heapsort, data_sorted, repetitions)
        t_quick_rand_s = _medir_tiempo(quicksort_random_pivot, data_sorted, repetitions)

        results['heapsort_sorted'].append(t_heap_s)
        results['quicksort_rand_pivot_sorted'].append(t_quick_rand_s)

    return results


def _medir_tiempo(func, data: list, repetitions: int) -> float:
    """Mide el tiempo promedio de ejecucion de una funcion."""
    tiempos = []
    for _ in range(repetitions):
        copia = data[:]
        inicio = time.perf_counter()
        func(copia)
        fin = time.perf_counter()
        tiempos.append(fin - inicio)
    return sum(tiempos) / len(tiempos)


def imprimir_comparativa(results: dict) -> None:
    """
    Imprime tabla comparativa y grafica ASCII de Heapsort vs Quicksort.
    """
    sizes = results['sizes']

    print("\n" + "=" * 90)
    print("  FASE 4: COMPARATIVA EMPIRICA - Heapsort vs Quicksort")
    print("=" * 90)

    print("\nAnalisis de Complejidad Teorica:")
    print("  Heapsort:  Big-O = O(n log n) | Big-Omega = Omega(n log n) | Theta = Theta(n log n)")
    print("  Quicksort: Big-O = O(n^2)     | Big-Omega = Omega(n log n) | Theta = Theta(n log n) promedio")
    print("  Quicksort (pivote aleatorio): Big-O esperado = O(n log n)")

    print("\nRecurrencias y Metodo Maestro:")
    print("  Heapsort - heapify: T(n) = T(n/2) + O(1)")
    print("    a=1, b=2, f(n)=1: n^log_2(1) = n^0 = 1 = Theta(1) -> Caso 2: T(n) = O(log n)")
    print("  Quicksort promedio: T(n) = 2T(n/2) + O(n)")
    print("    a=2, b=2, f(n)=n: n^log_2(2) = n^1 = Theta(n) -> Caso 2: T(n) = O(n log n)")
    print("  Quicksort peor: T(n) = T(n-1) + O(n) -> T(n) = O(n^2) [no aplica MM directamente]")

    print("\n" + "-" * 90)
    print(f"  {'N':>6} | {'Heapsort (rand)':>16} | {'Quicksort (rand)':>17} | {'QS pivot-rand':>14} | {'Heap (sorted)':>14}")
    print("-" * 90)

    for i, n in enumerate(sizes):
        th = results['heapsort_random'][i] * 1000
        tq = results['quicksort_random'][i] * 1000
        tqr = results['quicksort_rand_pivot_random'][i] * 1000
        ths = results['heapsort_sorted'][i] * 1000

        print(f"  {n:>6} | {th:>13.4f} ms | {tq:>14.4f} ms | {tqr:>11.4f} ms | {ths:>11.4f} ms")

    print("-" * 90)

    # Grafica ASCII de tiempo vs n (datos aleatorios)
    print("\n  Grafica: Tiempo vs N (datos aleatorios)")
    _grafica_ascii(sizes,
                   results['heapsort_random'],
                   results['quicksort_rand_pivot_random'],
                   "Heapsort", "QS-rand")


def _grafica_ascii(sizes: List[int],
                   tiempos1: List[float],
                   tiempos2: List[float],
                   label1: str,
                   label2: str,
                   width: int = 60,
                   height: int = 12) -> None:
    """
    Dibuja una grafica ASCII simple de dos series de datos.
    Eje X: tamanio n | Eje Y: tiempo en ms
    """
    all_times = tiempos1 + tiempos2
    max_t = max(all_times) if max(all_times) > 0 else 1
    min_t = 0

    print(f"\n  Tiempo (ms)  [{label1}=# , {label2}=*]")
    print(f"  {max_t*1000:.4f} |")

    for row in range(height, -1, -1):
        threshold = min_t + (max_t - min_t) * row / height
        line = f"  {'':>8} |"
        for i, n in enumerate(sizes):
            t1 = tiempos1[i]
            t2 = tiempos2[i]
            if abs(t1 - threshold) < (max_t - min_t) / height:
                line += " #"
            elif abs(t2 - threshold) < (max_t - min_t) / height:
                line += " *"
            else:
                line += "  "
        print(line)

    print("  " + "-" * (width + 12))
    label_line = "  N ->       |"
    for n in sizes:
        label_line += f" {n}"
    print(label_line)
