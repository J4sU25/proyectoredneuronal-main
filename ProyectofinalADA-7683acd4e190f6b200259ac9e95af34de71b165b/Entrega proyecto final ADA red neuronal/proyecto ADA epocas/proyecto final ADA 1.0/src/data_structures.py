"""
Estructuras de datos eficientes para optimización de redes neuronales
Incluye: MinHeap, MaxHeap, HashTable, Cola de Lotes
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
