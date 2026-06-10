"""
Estructuras de datos eficientes para optimizacion de redes neuronales
Incluye: MinHeap, MaxHeap, HashTable, Cola de Lotes, BST

Fase 2 (RA2): Modulos de seleccion/ordenamiento
  - quickselect_top_k: O(n) promedio, O(n^2) peor caso
  - quickselect_median: O(n) promedio via Quickselect
  - heap_top_k_vs_sort: Comparativa heap O(n log k) vs sort O(n log n)
  - MaxHeap top-k: Insert O(log k), Build O(n)

Fase 3 (RA3): Integracion de estructuras
  - BatchQueue: Cola de lotes O(1) enqueue/dequeue
  - HashTable: Hash de perdidas O(1) promedio insert/lookup
  - BST: Poda de pesos por magnitud O(log n) insert/search

Analisis de Recurrencias (Metodo Maestro):
  Quickselect: T(n) = T(n/2) + O(n) -> Caso 3 -> T(n) = O(n)
  Heapify:     T(n) = T(n/2) + O(1) -> Caso 2 -> T(n) = O(log n)
  Build-Heap:  Sum_{h=0}^{log n} (n/2^h) * O(h) = O(n)
"""

from typing import List, Tuple, Optional, Any, Callable


class MinHeap:
    """
    Heap mínimo para seleccionar top-k elementos
    Complejidad: Insert O(log n), Extract min O(log n), Build O(n)
    Espacio: O(n)
    """
    
    def __init__(self, items: Optional[List[Tuple[float, int]]] = None):
        """
        items: Lista de tuplas (valor, índice)
        Complejidad: O(n) con build-heap, O(n log n) si se insertan uno a uno
        """
        self.heap: List[Tuple[float, int]] = []
        
        if items:
            self.heap = items[:]
            self._build_heap()
    
    def _build_heap(self):
        """
        Construye el heap desde un arreglo
        Complejidad: O(n) - Método de Floyd
        """
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)
    
    def _heapify_up(self, idx: int):
        """
        Sube un elemento a su posición correcta
        Complejidad: O(log n)
        """
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] < self.heap[parent][0]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break
    
    def _heapify_down(self, idx: int):
        """
        Baja un elemento a su posición correcta
        Complejidad: O(log n)
        """
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            
            if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            
            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                idx = smallest
            else:
                break
    
    def insert(self, value: float, idx: int):
        """
        Inserta un elemento
        Complejidad: O(log n)
        """
        self.heap.append((value, idx))
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self) -> Tuple[float, int]:
        """
        Extrae el elemento mínimo
        Complejidad: O(log n)
        """
        if not self.heap:
            raise IndexError("Heap vacío")
        
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        
        if self.heap:
            self._heapify_down(0)
        
        return min_val
    
    def peek_min(self) -> Tuple[float, int]:
        """
        Retorna el mínimo sin extraer
        Complejidad: O(1)
        """
        if not self.heap:
            raise IndexError("Heap vacío")
        return self.heap[0]
    
    def is_empty(self) -> bool:
        """Complejidad: O(1)"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Complejidad: O(1)"""
        return len(self.heap)


class MaxHeap:
    """
    Heap máximo para seleccionar top-k elementos
    Complejidad: Insert O(log n), Extract max O(log n), Build O(n)
    """
    
    def __init__(self, items: Optional[List[Tuple[float, int]]] = None):
        self.heap: List[Tuple[float, int]] = []
        
        if items:
            self.heap = items[:]
            self._build_heap()
    
    def _build_heap(self):
        """Complejidad: O(n)"""
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)
    
    def _heapify_up(self, idx: int):
        """Complejidad: O(log n)"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] > self.heap[parent][0]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break
    
    def _heapify_down(self, idx: int):
        """Complejidad: O(log n)"""
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left][0] > self.heap[largest][0]:
                largest = left
            
            if right < len(self.heap) and self.heap[right][0] > self.heap[largest][0]:
                largest = right
            
            if largest != idx:
                self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
                idx = largest
            else:
                break
    
    def insert(self, value: float, idx: int):
        """Complejidad: O(log n)"""
        self.heap.append((value, idx))
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self) -> Tuple[float, int]:
        """Complejidad: O(log n)"""
        if not self.heap:
            raise IndexError("Heap vacío")
        
        max_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        
        if self.heap:
            self._heapify_down(0)
        
        return max_val
    
    def is_empty(self) -> bool:
        """Complejidad: O(1)"""
        return len(self.heap) == 0


class HashTable:
    """
    Tabla hash para almacenar pérdidas por índice de muestra
    Complejidad promedio: Insert O(1), Lookup O(1), Delete O(1)
    Complejidad caso peor: O(n) (colisiones)
    Espacio: O(n)
    """
    
    def __init__(self, capacity: int = 256):
        """
        Inicializa tabla hash con capacidad inicial
        Complejidad: O(capacity)
        """
        self.capacity = capacity
        self.table: List[List[Tuple[int, float]]] = [[] for _ in range(capacity)]
        self.size = 0
    
    def _hash(self, key: int) -> int:
        """
        Función hash
        Complejidad: O(1)
        """
        return key % self.capacity
    
    def insert(self, key: int, value: float):
        """
        Inserta o actualiza un par clave-valor
        Complejidad promedio: O(1)
        Complejidad caso peor: O(n) por colisiones
        """
        idx = self._hash(key)
        bucket = self.table[idx]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
        self.size += 1
    
    def lookup(self, key: int) -> Optional[float]:
        """
        Busca un valor por clave
        Complejidad promedio: O(1)
        Complejidad caso peor: O(n)
        """
        idx = self._hash(key)
        bucket = self.table[idx]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None
    
    def delete(self, key: int) -> bool:
        """
        Elimina un par clave-valor
        Complejidad promedio: O(1)
        Complejidad caso peor: O(n)
        """
        idx = self._hash(key)
        bucket = self.table[idx]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        
        return False
    
    def get_all(self) -> List[Tuple[int, float]]:
        """
        Retorna todos los pares clave-valor
        Complejidad: O(n + capacity)
        """
        result = []
        for bucket in self.table:
            result.extend(bucket)
        return result
    
    def clear(self):
        """
        Limpia la tabla
        Complejidad: O(capacity)
        """
        self.table = [[] for _ in range(self.capacity)]
        self.size = 0


class BatchQueue:
    """
    Cola para gestionar lotes (batches) de manera eficiente
    Complejidad: Enqueue O(1), Dequeue O(1), Peek O(1)
    Espacio: O(batch_size)
    """
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.queue: List[int] = []
    
    def enqueue(self, sample_idx: int):
        """
        Añade un índice de muestra a la cola
        Complejidad: O(1) amortizado
        """
        self.queue.append(sample_idx)
    
    def is_full(self) -> bool:
        """
        Verifica si el lote está completo
        Complejidad: O(1)
        """
        return len(self.queue) >= self.batch_size
    
    def dequeue_batch(self) -> List[int]:
        """
        Extrae un lote completo
        Complejidad: O(batch_size)
        """
        if len(self.queue) < self.batch_size:
            raise ValueError("Lote incompleto")
        
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        return batch
    
    def peek(self) -> List[int]:
        """
        Ve el lote sin extraerlo
        Complejidad: O(1)
        """
        return self.queue[:self.batch_size]
    
    def size(self) -> int:
        """Complejidad: O(1)"""
        return len(self.queue)
    
    def clear(self):
        """Complejidad: O(1)"""
        self.queue = []


def quickselect_top_k(losses: List[float], k: int) -> List[int]:
    """
    Encuentra los k índices con mayores pérdidas usando Quickselect
    Complejidad promedio: O(n)
    Complejidad caso peor: O(n²)
    Espacio: O(log n) en promedio por recursión
    """
    if k > len(losses):
        k = len(losses)
    
    indexed_losses = [(loss, idx) for idx, loss in enumerate(losses)]
    
    def partition(arr: List[Tuple[float, int]], low: int, high: int, pivot_idx: int) -> int:
        """
        Particiona el arreglo alrededor de un pivote
        Complejidad: O(n)
        """
        pivot_value = arr[pivot_idx][0]
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        store_idx = low
        for i in range(low, high):
            if arr[i][0] > pivot_value:  # Mayor que para top-k
                arr[i], arr[store_idx] = arr[store_idx], arr[i]
                store_idx += 1
        
        arr[store_idx], arr[high] = arr[high], arr[store_idx]
        return store_idx
    
    def select(arr: List[Tuple[float, int]], low: int, high: int, k_idx: int) -> int:
        """
        Selecciona el k-ésimo elemento
        Complejidad promedio: O(n)
        Complejidad caso peor: O(n²)
        """
        if low == high:
            return low
        
        # Elegir pivote aleatorio
        import random
        pivot_idx = random.randint(low, high)
        pivot_idx = partition(arr, low, high, pivot_idx)
        
        if k_idx == pivot_idx:
            return k_idx
        elif k_idx < pivot_idx:
            return select(arr, low, pivot_idx - 1, k_idx)
        else:
            return select(arr, pivot_idx + 1, high, k_idx)
    
    select(indexed_losses, 0, len(indexed_losses) - 1, k - 1)
    
    return [idx for loss, idx in indexed_losses[:k]]


# ============================================================
# QUICKSELECT MEDIANA - RA2
# Recurrencia: T(n) = T(n/2) + O(n) -> Metodo Maestro Caso 3 -> O(n)
# ============================================================

def quickselect_median(values: List[float]) -> float:
    """
    Encuentra la mediana de una lista usando Quickselect.

    Recurrencia de Quickselect:
    - Caso promedio (pivote aleatorio divide en mitades):
        T(n) = T(n/2) + O(n)
        Metodo Maestro: a=1, b=2, f(n)=n
        n^(log_b(a)) = n^0 = 1
        f(n) = n = Omega(n^(0+eps)) para eps=1 -> Caso 3
        Verificar condicion de regularidad: a*f(n/b) = n/2 <= c*n para c=0.5 < 1 [OK]
        => T(n) = Theta(n)

    - Caso peor (pivote siempre en extremo):
        T(n) = T(n-1) + O(n) = O(n^2)

    Big-O:    O(n^2) - Peor caso
    Big-Omega: Omega(n) - Mejor caso
    Big-Theta: Theta(n) - Caso promedio con pivote aleatorio

    Ventaja sobre sort: O(n) vs O(n log n) -> no hay que ordenar todo

    Complejidad temporal: O(n) promedio, O(n^2) peor caso
    Complejidad espacial: O(log n) promedio por pila de recursion
    """
    import random

    if not values:
        raise ValueError("Lista vacia")

    arr = [(v, i) for i, v in enumerate(values)]
    n = len(arr)
    target = n // 2  # Indice de la mediana (para n par, toma el menor)

    def partition_qs(a, low, high, pivot_idx):
        """Particion de Quickselect. Complejidad: O(n)"""
        pivot_val = a[pivot_idx][0]
        a[pivot_idx], a[high] = a[high], a[pivot_idx]
        store = low
        for i in range(low, high):
            if a[i][0] <= pivot_val:
                a[i], a[store] = a[store], a[i]
                store += 1
        a[store], a[high] = a[high], a[store]
        return store

    def select_qs(a, low, high, k_target):
        if low == high:
            return a[low][0]
        pivot_idx = random.randint(low, high)
        pivot_idx = partition_qs(a, low, high, pivot_idx)
        if k_target == pivot_idx:
            return a[k_target][0]
        elif k_target < pivot_idx:
            return select_qs(a, low, pivot_idx - 1, k_target)
        else:
            return select_qs(a, pivot_idx + 1, high, k_target)

    return select_qs(arr, 0, n - 1, target)


# ============================================================
# HEAP TOP-K VS SORT - RA2
# Comparativa empirica: heap O(n log k) vs sort O(n log n)
# Ventaja del heap cuando k << n
# ============================================================

def heap_top_k(losses: List[float], k: int) -> List[int]:
    """
    Encuentra los k indices con mayores perdidas usando Min-Heap de tamano k.

    Algoritmo:
    1. Insertar primeros k elementos en min-heap: O(k log k)
    2. Para cada elemento restante (n-k):
       - Si mayor que el minimo del heap, reemplazar: O(log k)
    3. Total: O(k log k + (n-k) log k) = O(n log k)

    Recurrencia del heapify: T(n) = T(n/2) + O(1) -> O(log n)
    Aplicado k veces para build: suma geometrica -> O(k)
    Luego n-k inserciones de O(log k) -> O(n log k)

    Big-O: O(n log k)
    Ventaja: Si k << n, es mucho mejor que O(n log n) del sort

    Complejidad temporal: O(n log k)
    Complejidad espacial: O(k) para el heap
    """
    import random as _random

    if k >= len(losses):
        return list(range(len(losses)))

    # Min-heap de tamano k para los k maximos
    # Almacena (perdida, indice)
    heap: List[Tuple[float, int]] = []

    def _hup(h, idx):
        """Heapify-up. O(log k)"""
        while idx > 0:
            p = (idx - 1) // 2
            if h[idx][0] < h[p][0]:
                h[idx], h[p] = h[p], h[idx]
                idx = p
            else:
                break

    def _hdown(h, idx, size):
        """Heapify-down. O(log k)"""
        while True:
            smallest = idx
            l, r = 2*idx+1, 2*idx+2
            if l < size and h[l][0] < h[smallest][0]:
                smallest = l
            if r < size and h[r][0] < h[smallest][0]:
                smallest = r
            if smallest != idx:
                h[idx], h[smallest] = h[smallest], h[idx]
                idx = smallest
            else:
                break

    # Construir heap inicial con primeros k elementos: O(k log k)
    for i in range(k):
        heap.append((losses[i], i))
        _hup(heap, len(heap) - 1)

    # Procesar elementos restantes: O((n-k) log k)
    for i in range(k, len(losses)):
        if losses[i] > heap[0][0]:  # Mayor que el minimo del heap
            heap[0] = (losses[i], i)
            _hdown(heap, 0, k)

    return [idx for loss, idx in heap]


def heap_top_k_vs_sort(losses: List[float], k: int) -> dict:
    """
    Compara heap_top_k vs sort para encontrar los k mayores.

    Analisis:
    - Heap top-k: O(n log k) tiempo, O(k) espacio
    - Sort + slice: O(n log n) tiempo, O(n) espacio

    Cuando k << n: heap es mas eficiente
    Cuando k ~ n:  sort puede ser comparable por constantes

    Recurrencia relevante: ambos usan heapify internamente
    heapify: T(n) = T(n/2) + O(1) -> Caso 2 MM -> O(log n)

    Complejidad temporal: O(n log k) para heap, O(n log n) para sort
    Complejidad espacial: O(k) para heap, O(n) para sort
    """
    import time as _time

    # Metodo 1: Heap
    inicio_heap = _time.perf_counter()
    resultado_heap = heap_top_k(losses, k)
    tiempo_heap = _time.perf_counter() - inicio_heap

    # Metodo 2: Sort completo + slice
    inicio_sort = _time.perf_counter()
    indexed = list(enumerate(losses))
    indexed_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)
    resultado_sort = [idx for idx, _ in indexed_sorted[:k]]
    tiempo_sort = _time.perf_counter() - inicio_sort

    # Verificar que ambos tienen los mismos elementos (en cualquier orden)
    coinciden = set(resultado_heap) == set(resultado_sort)

    return {
        'tiempo_heap': tiempo_heap,
        'tiempo_sort': tiempo_sort,
        'speedup_heap': tiempo_sort / tiempo_heap if tiempo_heap > 0 else float('inf'),
        'resultado_heap': sorted(resultado_heap),
        'resultado_sort': sorted(resultado_sort),
        'coinciden': coinciden,
        'n': len(losses),
        'k': k
    }


def imprimir_analisis_ra2(n_values=None, k_values=None) -> None:
    """
    Ejecuta y muestra la comparativa empirica heap vs sort para varios n y k.
    Valida el analisis teorico RA2.
    """
    import random as _random
    import time as _time

    if n_values is None:
        n_values = [100, 500, 1000, 5000]
    if k_values is None:
        k_values = [10, 50]

    print("\n" + "=" * 90)
    print("  RA2: COMPARATIVA HEAP top-k vs SORT - Mediana con Quickselect")
    print("=" * 90)

    print("\nAnalisis Teorico:")
    print("  Heap top-k:  O(n log k) - optimo para k << n")
    print("  Sort + slice: O(n log n) - independiente de k")
    print("  Speedup teorico esperado: O(n log n) / O(n log k) = log(n)/log(k)")

    print("\nMetodo Maestro para Quickselect:")
    print("  T(n) = T(n/2) + O(n)")
    print("  a=1, b=2, f(n)=n")
    print("  n^log_b(a) = n^0 = 1")
    print("  f(n) = n = Omega(n^(0+1)) -> Caso 3 del Metodo Maestro")
    print("  Condicion regularidad: 1*f(n/2) = n/2 <= c*n, c=1/2 < 1 [VERIFICADO]")
    print("  => T(n) = Theta(n)")

    print("\n  Mediana via Quickselect: O(n) promedio vs O(n log n) si se ordena todo")
    print("  Ventaja: Para encontrar solo la mediana, no hace falta ordenar todo")

    print("\n" + "-" * 90)
    print(f"  {'n':>6} | {'k':>5} | {'Heap (ms)':>10} | {'Sort (ms)':>10} | {'Speedup':>8} | {'Coinciden':>10}")
    print("-" * 90)

    for n in n_values:
        losses = [_random.uniform(0, 100) for _ in range(n)]
        for k in k_values:
            if k >= n:
                continue
            r = heap_top_k_vs_sort(losses, k)
            print(f"  {n:>6} | {k:>5} | {r['tiempo_heap']*1000:>10.4f} | "
                  f"{r['tiempo_sort']*1000:>10.4f} | {r['speedup_heap']:>8.2f}x | "
                  f"{'Si' if r['coinciden'] else 'NO':>10}")

    print("-" * 90)
    print("  Speedup > 1 significa que heap es mas rapido que sort")
    print("  A medida que n crece y k se mantiene pequeno, la ventaja del heap aumenta")


# ============================================================
# BST (Binary Search Tree) - Fase 3 (RA3)
# Para poda de pesos por magnitud
# ============================================================

class BSTNode:
    """Nodo del arbol BST."""
    def __init__(self, magnitude: float, layer_idx: int, i: int, j: int):
        self.magnitude = magnitude  # |peso| usado como clave
        self.layer_idx = layer_idx
        self.i = i  # fila del peso en la matriz
        self.j = j  # columna del peso en la matriz
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None


class BST:
    """
    Arbol Binario de Busqueda para poda de pesos por magnitud.

    Los pesos se indexan por |magnitud|. La poda elimina los pesos
    con menor magnitud (los mas cercanos a cero), que contribuyen poco.

    Complejidad temporal (arbol balanceado):
    - Insert:   O(log n)
    - Search:   O(log n)
    - Delete:   O(log n)
    - Min/Max:  O(log n)
    - Inorder:  O(n)

    Complejidad temporal (arbol degenerado - peor caso):
    - Insert, Search, Delete: O(n)
    => En la practica, los pesos de una red neuronal tienen distribucion
       aproximadamente gaussiana, por lo que el arbol tiende a estar
       razonablemente balanceado.

    Complejidad espacial: O(n) donde n = numero de pesos

    Recurrencia busqueda (arbol balanceado):
    T(n) = T(n/2) + O(1)
    Metodo Maestro: a=1, b=2, f(n)=1
    n^log_2(1) = 1 = Theta(1) = f(n) -> Caso 2
    => T(n) = O(log n)
    """

    def __init__(self):
        self.root: Optional[BSTNode] = None
        self.size: int = 0

    def insert(self, magnitude: float, layer_idx: int, i: int, j: int) -> None:
        """
        Inserta un peso en el BST ordenado por magnitud.

        Complejidad: O(log n) promedio, O(n) peor caso
        """
        nuevo = BSTNode(magnitude, layer_idx, i, j)
        self.size += 1

        if self.root is None:
            self.root = nuevo
            return

        actual = self.root
        while True:
            if magnitude <= actual.magnitude:
                if actual.left is None:
                    actual.left = nuevo
                    break
                actual = actual.left
            else:
                if actual.right is None:
                    actual.right = nuevo
                    break
                actual = actual.right

    def find_min(self) -> Optional[BSTNode]:
        """
        Encuentra el nodo con menor magnitud (mas proximo a cero).

        Complejidad: O(log n) balanceado, O(n) peor caso
        """
        if self.root is None:
            return None
        actual = self.root
        while actual.left is not None:
            actual = actual.left
        return actual

    def find_max(self) -> Optional[BSTNode]:
        """
        Encuentra el nodo con mayor magnitud.

        Complejidad: O(log n)
        """
        if self.root is None:
            return None
        actual = self.root
        while actual.right is not None:
            actual = actual.right
        return actual

    def get_smallest_k(self, k: int) -> List[BSTNode]:
        """
        Retorna los k pesos de menor magnitud (candidatos a podar).

        Recorre el arbol en inorder (izquierda -> raiz -> derecha)
        que produce los elementos en orden ascendente de magnitud.

        Complejidad: O(k + log n) usando recorrido inorder parcial
        Complejidad espacial: O(k + log n) por pila de recursion
        """
        resultado: List[BSTNode] = []

        def inorder(nodo: Optional[BSTNode]) -> bool:
            if nodo is None or len(resultado) >= k:
                return len(resultado) >= k
            if inorder(nodo.left):
                return True
            resultado.append(nodo)
            if len(resultado) >= k:
                return True
            return inorder(nodo.right)

        inorder(self.root)
        return resultado

    def inorder_list(self) -> List[BSTNode]:
        """
        Retorna todos los nodos en orden ascendente de magnitud.

        Complejidad: O(n)
        Complejidad espacial: O(n + log n) = O(n)
        """
        resultado: List[BSTNode] = []

        def _inorder(nodo):
            if nodo is None:
                return
            _inorder(nodo.left)
            resultado.append(nodo)
            _inorder(nodo.right)

        _inorder(self.root)
        return resultado


def construir_bst_desde_modelo(model) -> BST:
    """
    Construye un BST indexando todos los pesos del modelo por magnitud.

    Complejidad temporal: O(P log P) donde P = total de parametros
    Complejidad espacial: O(P) para el BST
    """
    bst = BST()
    for layer_idx, layer in enumerate(model.layers):
        for i in range(layer.W.rows):
            for j in range(layer.W.cols):
                magnitude = abs(layer.W.data[i][j])
                bst.insert(magnitude, layer_idx, i, j)
    return bst


def podar_pesos_bst(model, umbral: float = 0.01) -> dict:
    """
    Poda (pone a cero) pesos cuya magnitud es menor al umbral.
    Usa BST para identificar eficientemente los pesos a podar.

    Estrategia de poda:
    - Los pesos cercanos a cero contribuyen poco al aprendizaje
    - Podarlos reduce complejidad del modelo y puede prevenir overfitting
    - Trade-off: mas poda = mas compresion pero posible perdida de precision

    Complejidad: O(P log P) para construir BST + O(P_podados) para podar
    donde P_podados <= P son los pesos menores al umbral.

    Retorna: estadisticas de poda
    """
    bst = construir_bst_desde_modelo(model)
    total_pesos = bst.size
    pesos_podados = 0

    # Recorrer inorder y podar los que esten por debajo del umbral
    for nodo in bst.inorder_list():
        if nodo.magnitude < umbral:
            model.layers[nodo.layer_idx].W.data[nodo.i][nodo.j] = 0.0
            pesos_podados += 1
        else:
            break  # Los demas son mayores, no hace falta seguir

    return {
        'total_pesos': total_pesos,
        'pesos_podados': pesos_podados,
        'ratio_poda': pesos_podados / total_pesos if total_pesos > 0 else 0,
        'umbral': umbral
    }


def imprimir_analisis_ra3(model) -> None:
    """
    Muestra el impacto de las estructuras de datos en RA3.
    Reporta BST para poda, uso de HashTable y BatchQueue.
    """
    print("\n" + "=" * 80)
    print("  RA3: ANALISIS DE IMPACTO DE ESTRUCTURAS DE DATOS")
    print("=" * 80)

    # BST - Poda de pesos
    print("\n  [BST] Analisis de poda de pesos por magnitud:")
    bst = construir_bst_desde_modelo(model)
    nodos = bst.inorder_list()

    if nodos:
        min_mag = nodos[0].magnitude
        max_mag = nodos[-1].magnitude
        # Estimacion de cuantos pesos tendrian magnitud < varios umbrales
        umbrales = [0.001, 0.005, 0.01, 0.05]
        print(f"    Pesos totales en BST: {bst.size:,}")
        print(f"    Magnitud minima: {min_mag:.6f}")
        print(f"    Magnitud maxima: {max_mag:.6f}")
        print(f"\n    Pesos que se podarian por umbral:")
        for u in umbrales:
            conteo = sum(1 for n in nodos if n.magnitude < u)
            ratio = conteo / bst.size * 100
            print(f"      Umbral {u:.3f}: {conteo:,} pesos ({ratio:.1f}%)")

    print("\n  [HashTable] Uso para cache de perdidas:")
    print("    Complejidad insert/lookup: O(1) promedio")
    print("    Permite recordar la perdida de cada muestra entre epocas")
    print("    Util para Hard Mining: solo re-entrenar con muestras de alta perdida")

    print("\n  [BatchQueue] Cola de lotes:")
    print("    Complejidad enqueue/dequeue: O(1)")
    print("    Gestiona el flujo de mini-batches de forma eficiente")
    print("    Evita recalcular indices de batch en cada epoca")

    print("\n  [MinHeap/MaxHeap] Seleccion top-k:")
    print("    Complejidad build: O(n), insert: O(log n), extract: O(log n)")
    print("    Para Hard Mining: seleccionar top-k muestras dificiles")
    print("    O(n log k) total vs O(n log n) con sort -> ventaja para k << n")
