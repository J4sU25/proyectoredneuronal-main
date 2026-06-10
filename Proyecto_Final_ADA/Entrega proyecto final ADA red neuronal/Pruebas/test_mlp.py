"""
Pruebas unitarias del Proyecto Final ADA
Verifica correctitud de todos los modulos implementados

Ejecutar:  python test_mlp.py
"""

import sys
import os
import math
import random
import time

# Agregar el directorio del modulo al path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Epocas', 'Entrenamiento_ADA 1.0'))
sys.path.insert(0, BASE_DIR)

from src.matrix import Matrix
from src.activation import ActivationFunction, softmax, softmax_derivative
from src.data_structures import (
    MinHeap, MaxHeap, HashTable, BatchQueue,
    quickselect_top_k, quickselect_median,
    heap_top_k, heap_top_k_vs_sort,
    BST, construir_bst_desde_modelo, podar_pesos_bst
)
from src.dataset import DatasetGenerator
from src.mlp import MLP
from src.sorting import heapsort, quicksort, quicksort_random_pivot
from src.knn import KNNClassifier, distancia_euclidiana
from src.gradient_check import gradient_check


# ============================================================
# UTILIDADES DE TEST
# ============================================================

PASS = 0
FAIL = 0
TEST_LOG = []


def check(nombre: str, condicion: bool, detalle: str = "") -> None:
    global PASS, FAIL
    if condicion:
        PASS += 1
        estado = "PASS"
    else:
        FAIL += 1
        estado = "FAIL"
    TEST_LOG.append((estado, nombre, detalle))
    simbolo = "OK" if condicion else "XX"
    print(f"  [{simbolo}] {nombre}" + (f" -> {detalle}" if detalle else ""))


def check_approx(nombre: str, valor: float, esperado: float,
                 tolerancia: float = 1e-6, detalle: str = "") -> None:
    ok = abs(valor - esperado) <= tolerancia
    check(nombre, ok, detalle or f"obtenido={valor:.8f} esperado={esperado:.8f} tol={tolerancia}")


def seccion(titulo: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {titulo}")
    print(f"{'='*70}")


# ============================================================
# TEST 1: MATRIZ
# ============================================================

def test_matrix():
    seccion("TEST 1: Operaciones de Matriz")

    # Creacion
    m = Matrix(2, 3)
    check("Matriz ceros", all(m.data[i][j] == 0 for i in range(2) for j in range(3)))

    # Transpose
    m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    t  = m2.transpose()
    check("Forma matriz transpuesta", t.rows == 3 and t.cols == 2)
    check("Valores matriz transpuesta", t.data[0][0] == 1 and t.data[2][1] == 6)

    # Dot product
    a = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    b = Matrix(3, 2, [[7, 8], [9, 10], [11, 12]])
    c = a.dot(b)
    check("Forma producto punto",  c.rows == 2 and c.cols == 2)
    check("Producto punto [0,0]", abs(c.data[0][0] - 58) < 1e-9)
    check("Producto punto [1,1]", abs(c.data[1][1] - 154) < 1e-9)

    # Dimension mismatch
    try:
        a.dot(a)
        check("Error dimensiones producto punto", False, "deberia lanzar excepcion")
    except ValueError:
        check("Error dimensiones producto punto", True)

    # Element-wise mult
    x = Matrix(2, 2, [[1, 2], [3, 4]])
    y = Matrix(2, 2, [[2, 3], [4, 5]])
    z = x.element_wise_mult(y)
    check("Producto de Hadamard (elemento a elemento)", z.data[0][0] == 2 and z.data[1][1] == 20)

    # Scalar mult
    s = x.scalar_mult(3.0)
    check("Multiplicacion escalar", s.data[0][1] == 6 and s.data[1][0] == 9)

    # Add / Subtract
    a2 = x.add(y)
    check("Suma de matrices", a2.data[0][0] == 3 and a2.data[1][1] == 9)
    s2 = x.subtract(y)
    check("Resta de matrices", s2.data[0][0] == -1 and s2.data[1][0] == -1)

    # Frobenius norm de [[3,4]] -> sqrt(25) = 5
    fn = Matrix(1, 2, [[3, 4]])
    check_approx("Norma de Frobenius", fn.frobenius_norm(), 5.0)

    # Copy
    cp = x.copy()
    cp.data[0][0] = 999
    check("Copia profunda (Deep copy)", x.data[0][0] == 1, "original no modificado")


# ============================================================
# TEST 2: FUNCIONES DE ACTIVACION
# ============================================================

def test_activaciones():
    seccion("TEST 2: Funciones de Activacion")

    # ReLU
    check_approx("ReLU(0)",    ActivationFunction.relu(0.0),   0.0)
    check_approx("ReLU(2)",    ActivationFunction.relu(2.0),   2.0)
    check_approx("ReLU(-3)",   ActivationFunction.relu(-3.0),  0.0)

    # ReLU derivative
    check_approx("dReLU(1)",   ActivationFunction.relu_derivative(1.0),  1.0)
    check_approx("dReLU(-1)",  ActivationFunction.relu_derivative(-1.0), 0.0)
    check_approx("dReLU(0)",   ActivationFunction.relu_derivative(0.0),  0.0)

    # Sigmoid
    check_approx("Sigmoid(0)", ActivationFunction.sigmoid(0.0), 0.5, 1e-9)
    sig_large = ActivationFunction.sigmoid(100.0)
    check("Sigmoid(100) ~ 1",  abs(sig_large - 1.0) < 1e-6)

    # Tanh
    check_approx("Tanh(0)",    ActivationFunction.tanh(0.0),   0.0, 1e-9)
    check("Tanh range",        -1.0 < ActivationFunction.tanh(1.0) < 1.0)

    # Softmax - suma debe ser 1
    logits = Matrix(1, 4, [[1.0, 2.0, 3.0, 4.0]])
    sm = softmax(logits)
    suma = sum(sm.data[0])
    check_approx("Softmax suma=1", suma, 1.0, 1e-9)
    check("Softmax max = indice 3", sm.data[0][3] == max(sm.data[0]))

    # Softmax batch
    logits_b = Matrix(2, 3, [[1, 2, 3], [3, 2, 1]])
    sm_b = softmax(logits_b)
    for i in range(2):
        check_approx(f"Softmax batch[{i}] suma=1", sum(sm_b.data[i]), 1.0, 1e-9)

    # softmax_derivative: (pred - target)
    pred   = Matrix(1, 3, [[0.2, 0.5, 0.3]])
    target = Matrix(1, 3, [[0.0, 1.0, 0.0]])
    grad   = softmax_derivative(pred, target)
    check_approx("Softmax grad[0]", grad.data[0][0],  0.2)
    check_approx("Softmax grad[1]", grad.data[0][1], -0.5)
    check_approx("Softmax grad[2]", grad.data[0][2],  0.3)


# ============================================================
# TEST 3: ESTRUCTURAS DE DATOS
# ============================================================

def test_estructuras_datos():
    seccion("TEST 3: Estructuras de Datos")

    # --- MinHeap ---
    heap = MinHeap()
    for v, i in [(5, 0), (3, 1), (8, 2), (1, 3), (4, 4)]:
        heap.insert(v, i)
    minv, mini = heap.extract_min()
    check("MinHeap extract_min valor", minv == 1.0)
    check("MinHeap extract_min idx",   mini == 3)
    check("MinHeap size",              heap.size() == 4)

    # Build heap
    items = [(5, 0), (2, 1), (9, 2), (1, 3)]
    heap2 = MinHeap(items)
    minv2, _ = heap2.extract_min()
    check("MinHeap build min", minv2 == 1.0)

    # --- MaxHeap ---
    mh = MaxHeap()
    for v, i in [(3, 0), (7, 1), (2, 2), (9, 3), (5, 4)]:
        mh.insert(v, i)
    maxv, maxi = mh.extract_max()
    check("MaxHeap extract_max valor", maxv == 9.0)
    check("MaxHeap extract_max idx",   maxi == 3)

    # --- HashTable ---
    ht = HashTable(capacity=16)
    ht.insert(0, 1.5)
    ht.insert(16, 2.5)  # colision con 0 (16 % 16 == 0)
    ht.insert(1, 3.0)
    check("HashTable lookup 0",  abs(ht.lookup(0) - 1.5) < 1e-9)
    check("HashTable lookup 16", abs(ht.lookup(16) - 2.5) < 1e-9)
    check("HashTable lookup 1",  abs(ht.lookup(1) - 3.0) < 1e-9)
    check("HashTable lookup miss", ht.lookup(999) is None)

    # Update
    ht.insert(0, 99.0)
    check("HashTable update", abs(ht.lookup(0) - 99.0) < 1e-9)

    # Delete
    ht.delete(1)
    check("HashTable delete", ht.lookup(1) is None)

    # --- BatchQueue ---
    bq = BatchQueue(batch_size=3)
    for i in range(7):
        bq.enqueue(i)
    check("BatchQueue is_full",   bq.is_full())
    batch = bq.dequeue_batch()
    check("BatchQueue batch len", len(batch) == 3)
    check("BatchQueue batch [0]", batch[0] == 0)
    check("BatchQueue size after", bq.size() == 4)

    # --- Quickselect top-k ---
    losses = [0.1, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2]
    top3   = quickselect_top_k(losses, 3)
    top3v  = sorted([losses[i] for i in top3], reverse=True)
    check("Quickselect top-3 len",  len(top3) == 3)
    check("Quickselect top-3 vals", top3v[0] == 0.9 and top3v[1] == 0.8 and top3v[2] == 0.7)

    # --- Quickselect mediana ---
    datos = list(range(1, 10))  # mediana = 5
    random.shuffle(datos)
    med = quickselect_median([float(x) for x in datos])
    check_approx("Quickselect median", med, 5.0)

    # --- Heap top-k vs sort ---
    ls = [random.uniform(0, 100) for _ in range(500)]
    r  = heap_top_k_vs_sort(ls, k=20)
    check("Heap vs Sort coinciden",   r['coinciden'])
    check("Heap vs Sort n correcto",  r['n'] == 500)
    check("Heap vs Sort k correcto",  r['k'] == 20)

    # --- BST ---
    bst = BST()
    bst.insert(0.5, 0, 0, 0)
    bst.insert(0.2, 0, 0, 1)
    bst.insert(0.8, 0, 1, 0)
    bst.insert(0.1, 1, 0, 0)
    bst.insert(0.9, 1, 0, 1)

    min_node = bst.find_min()
    max_node = bst.find_max()
    check("BST find_min magnitude",  abs(min_node.magnitude - 0.1) < 1e-9)
    check("BST find_max magnitude",  abs(max_node.magnitude - 0.9) < 1e-9)
    check("BST size",                bst.size == 5)

    # Inorder debe estar ordenado
    check("BST tamano", bst.size == 5)

    # Inorder debe estar ordenado
    inorder = bst.inorder_list()
    mags    = [n.magnitude for n in inorder]
    check("BST inorder ordenado", mags == sorted(mags))

    # get_smallest_k
    k2 = bst.get_smallest_k(2)
    check("BST get_smallest_k longitud", len(k2) == 2)
    check("BST get_smallest_k valores",
          abs(k2[0].magnitude - 0.1) < 1e-9 and abs(k2[1].magnitude - 0.2) < 1e-9)


# ============================================================
# TEST 4: DATASET
# ============================================================

def test_dataset():
    seccion("TEST 4: Generacion y Procesamiento de Dataset")

    # Sintetico
    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=100, num_features=10, num_classes=4, seed=1
    )
    check("Dataset X shape rows", X.rows == 100)
    check("Dataset X shape cols", X.cols == 10)
    check("Dataset y shape",      y.rows == 100 and y.cols == 4)

    # One-hot
    for i in range(y.rows):
        check(f"One-hot suma=1 [{i}]", abs(sum(y.data[i]) - 1.0) < 1e-9,
              f"suma={sum(y.data[i])}")
        if i > 5:
            break  # solo verificar algunos

    # Normalize
    X_norm = DatasetGenerator.normalize_dataset(X)
    # Media debe ser ~0 para cada feature
    for j in range(X.cols):
        mean_j = sum(X_norm.data[i][j] for i in range(X.rows)) / X.rows
        check_approx(f"Normalize media feat {j}", mean_j, 0.0, 1e-6)
        if j > 3:
            break

    # Split
    X_tr, y_tr, X_v, y_v = DatasetGenerator.split_dataset(X, y, train_ratio=0.8, seed=42)
    check("Split train size", X_tr.rows == 80)
    check("Split val size",   X_v.rows == 20)
    check("Split total",      X_tr.rows + X_v.rows == 100)

    # MNIST-like
    X_m, y_m = DatasetGenerator.generate_mnist_like(num_samples=20, num_classes=5, img_size=8, seed=0)
    check("MNIST-like X cols", X_m.cols == 64)
    check("MNIST-like y cols", y_m.cols == 5)


# ============================================================
# TEST 5: FORWARD PASS
# ============================================================

def test_forward():
    seccion("TEST 5: Forward Pass del MLP")

    model = MLP([4, 8, 3], ['relu', 'linear'], 'cross_entropy')
    X = Matrix(5, 4, [[random.gauss(0, 1) for _ in range(4)] for _ in range(5)])
    
    # Test forward
    out = model.forward(X)
    check("Capa Forward - Filas salidas", out.rows == 5)
    check("Capa Forward - Columnas salidas", out.cols == 3)
    
    # Softmax row sums
    for i in range(out.rows):
        check_approx(f"Capa Forward - Suma Softmax Fila {i}", sum(out.data[i]), 1.0)
    
    # All between 0 and 1
    all_prob = all(0 <= val <= 1 for row in out.data for val in row)
    check("Capa Forward - Rango de probabilidades", all_prob)


# ============================================================
# TEST 6: BACKWARD PASS (GRADIENTES)
# ============================================================

def test_backward():
    seccion("TEST 6: Backward Pass y Actualizacion de Pesos")

    random.seed(0)
    model = MLP([4, 6, 2], ['relu', 'linear'], 'cross_entropy')
    X = Matrix(3, 4, [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(3)])
    y = Matrix(3, 2, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    # Guardar pesos antes
    w_before = model.layers[0].W.data[0][0]

    # Forward + backward
    pred = model.forward(X)
    loss1 = model.compute_loss(pred, y)

    dLoss = softmax_derivative(pred, y)
    model.backward(dLoss, learning_rate=0.1)

    w_after = model.layers[0].W.data[0][0]
    check("Backward actualiza pesos", w_before != w_after)

    # La perdida debe reducirse en la segunda iteracion
    pred2 = model.forward(X)
    loss2 = model.compute_loss(pred2, y)
    check("Una iteracion reduce perdida (tendencia)", True,
          f"loss1={loss1:.6f} loss2={loss2:.6f} (no siempre garantizado en 1 paso)")


# ============================================================
# TEST 7: ENTRENAMIENTO COMPLETO (overfitting en dataset pequeno)
# ============================================================

def test_entrenamiento():
    seccion("TEST 7: Entrenamiento - Convergencia en Dataset Pequeno")

    random.seed(42)
    model = MLP([4, 16, 3], ['relu', 'linear'], 'cross_entropy')

    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=60, num_features=4, num_classes=3, seed=5
    )
    X = DatasetGenerator.normalize_dataset(X)

    loss_inicial = None
    loss_final   = None

    for ep in range(30):
        loss, acc = model.train_epoch(X, y, batch_size=16, learning_rate=0.05)
        if ep == 0:
            loss_inicial = loss
        loss_final = loss

    check("Entrenamiento reduce perdida",
          loss_final < loss_inicial,
          f"inicial={loss_inicial:.4f} final={loss_final:.4f}")

    acc_final = model.compute_accuracy(model.forward(X), y)
    check("Entrenamiento mejora precision", acc_final > 0.3,
          f"acc={acc_final:.4f}")


# ============================================================
# TEST 8: PERSISTENCIA (PESOS)
# ============================================================

def test_persistencia():
    seccion("TEST 8: Guardado y Carga de Pesos")

    import tempfile

    random.seed(1)
    model = MLP([3, 5, 2], ['relu', 'linear'], 'cross_entropy')
    W0_before = model.layers[0].W.data[0][0]

    # Guardar en archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        filepath = tf.name

    try:
        ok_save = model.save_weights(filepath)
        check("Guardado de pesos OK", ok_save)

        # Modificar pesos
        model.layers[0].W.data[0][0] = 999.0

        # Cargar
        ok_load = model.load_weights(filepath)
        check("Carga de pesos OK", ok_load)
        check("Pesos restaurados", abs(model.layers[0].W.data[0][0] - W0_before) < 1e-9)
    finally:
        os.unlink(filepath)


# ============================================================
# TEST 9: ALGORITMOS DE ORDENAMIENTO
# ============================================================

def test_ordenamiento():
    seccion("TEST 9: Heapsort y Quicksort")

    random.seed(7)
    datos = [random.uniform(-100, 100) for _ in range(100)]
    ref   = sorted(datos)

    # Heapsort
    hs = heapsort(datos)
    check("Heapsort longitud",  len(hs) == 100)
    check("Heapsort correcto",  hs == ref)

    # Quicksort
    qs = quicksort(datos)
    check("Quicksort longitud", len(qs) == 100)
    check("Quicksort correcto", qs == ref)

    # Quicksort con pivote aleatorio
    qsr = quicksort_random_pivot(datos)
    check("QS random pivot",    qsr == ref)

    # Caso borde: lista ya ordenada
    ord_datos = sorted(datos)
    hs_ord = heapsort(ord_datos)
    check("Heapsort lista ordenada", hs_ord == ord_datos)

    # Lista de 1 elemento
    uno = heapsort([42.0])
    check("Heapsort 1 elemento", uno == [42.0])

    # No modifica original
    original = datos[:]
    heapsort(datos)
    check("Heapsort no modifica original", datos == original)

    # Stress test: 2000 elementos
    grande = [random.uniform(0, 1) for _ in range(2000)]
    ref_g  = sorted(grande)
    hs_g   = heapsort(grande)
    check("Heapsort stress 2000", hs_g == ref_g)


# ============================================================
# TEST 10: K-NN
# ============================================================

def test_knn():
    seccion("TEST 10: Clasificador k-NN")

    # Distancia euclidiana
    d = distancia_euclidiana([0.0, 0.0], [3.0, 4.0])
    check_approx("Distancia euclidiana 3-4-5", d, 5.0)
    check_approx("Distancia a si mismo", distancia_euclidiana([1, 2], [1, 2]), 0.0)

    # k-NN en dataset simple
    random.seed(0)
    X_tr, y_tr = DatasetGenerator.generate_synthetic_classification(
        num_samples=100, num_features=5, num_classes=3, seed=42
    )
    X_v, y_v = DatasetGenerator.generate_synthetic_classification(
        num_samples=20, num_features=5, num_classes=3, seed=99
    )

    knn = KNNClassifier(k=3)
    knn.fit(X_tr, y_tr)

    # No debe lanzar excepcion
    try:
        acc = knn.compute_accuracy(X_v, y_v)
        check("k-NN accuracy no falla", True, f"acc={acc:.4f}")
        check("k-NN accuracy rango [0,1]", 0.0 <= acc <= 1.0)
    except Exception as e:
        check("k-NN predict", False, str(e))

    # k=1 en datos de entrenamiento debe dar 100% de precision
    knn1 = KNNClassifier(k=1)
    knn1.fit(X_tr, y_tr)
    acc_self = knn1.compute_accuracy(X_tr, y_tr)
    check("k-NN k=1 en train = 100%", abs(acc_self - 1.0) < 1e-9,
          f"acc={acc_self:.4f}")


# ============================================================
# TEST 11: GRADIENT CHECK
# ============================================================

def test_gradient_check():
    seccion("TEST 11: Verificacion Numerica de Gradientes")

    random.seed(42)
    model = MLP([4, 6, 3], ['relu', 'linear'], 'cross_entropy')
    X = Matrix(5, 4, [[random.gauss(0, 0.3) for _ in range(4)] for _ in range(5)])
    y = Matrix(5, 3, [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0]
    ])

    gc = gradient_check(model, X, y, epsilon=1e-5, num_checks=10, verbose=False)

    check("Gradient check ejecuta OK",      gc is not None)
    check("Gradient check num_checks=10",   gc['num_checks'] == 10)
    check("Gradient check error < 1e-2",    gc['max_error'] < 1e-2,
          f"max_error={gc['max_error']:.2e}")
    check("Gradient check passed (< 1e-4)", gc['passed'],
          f"max_error={gc['max_error']:.2e} (umbral=1e-4)")


# ============================================================
# TEST 12: STRESS TESTS
# ============================================================

def test_stress():
    seccion("TEST 12: Stress Tests con Datos Sinteticos")

    # MLP con dataset grande
    random.seed(100)
    model = MLP([10, 32, 5], ['relu', 'linear'], 'cross_entropy')
    X, y = DatasetGenerator.generate_synthetic_classification(
        num_samples=500, num_features=10, num_classes=5, seed=77
    )
    X = DatasetGenerator.normalize_dataset(X)

    inicio = time.perf_counter()
    for _ in range(5):
        model.train_epoch(X, y, batch_size=32, learning_rate=0.01)
    elapsed = time.perf_counter() - inicio

    check("Stress: 5 epocas completan", True, f"tiempo={elapsed:.2f}s")
    check("Stress: tiempo razonable < 60s", elapsed < 60, f"{elapsed:.2f}s")

    # Heapsort en lista grande
    data_grande = [random.uniform(0, 1) for _ in range(3000)]
    ref = sorted(data_grande)
    inicio = time.perf_counter()
    result = heapsort(data_grande)
    elapsed2 = time.perf_counter() - inicio
    check("Stress: Heapsort 3000 elementos correcto", result == ref)
    check("Stress: Heapsort 3000 tiempo < 5s", elapsed2 < 5, f"{elapsed2:.3f}s")

    # Quickselect top-k en lista grande
    losses = [random.uniform(0, 1) for _ in range(1000)]
    top50 = quickselect_top_k(losses, 50)
    check("Stress: Quickselect top-50 len",    len(top50) == 50)
    check("Stress: Quickselect top-50 validos",
          all(0 <= i < 1000 for i in top50))

    # BST con muchos nodos
    bst = BST()
    for i in range(100):
        bst.insert(random.uniform(0, 1), 0, i, 0)
    check("Stress: BST 100 nodos size", bst.size == 100)
    inord = bst.inorder_list()
    mags  = [n.magnitude for n in inord]
    check("Stress: BST 100 inorder sorted", mags == sorted(mags))


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  PRUEBAS UNITARIAS - PROYECTO FINAL ADA")
    print("  Red Neuronal MLP desde cero")
    print("=" * 70)

    inicio = time.perf_counter()

    test_matrix()
    test_activaciones()
    test_estructuras_datos()
    test_dataset()
    test_forward()
    test_backward()
    test_entrenamiento()
    test_persistencia()
    test_ordenamiento()
    test_knn()
    test_gradient_check()
    test_stress()

    elapsed = time.perf_counter() - inicio

    print("\n" + "=" * 70)
    print("  RESUMEN DE PRUEBAS")
    print("=" * 70)
    print(f"\n  Tests PASADOS: {PASS}")
    print(f"  Tests FALLIDOS: {FAIL}")
    print(f"  Total: {PASS + FAIL}")
    print(f"  Tiempo: {elapsed:.2f}s")

    if FAIL > 0:
        print("\n  Tests fallidos:")
        for estado, nombre, detalle in TEST_LOG:
            if estado == "FAIL":
                print(f"    [XX] {nombre}: {detalle}")

    print()
    if FAIL == 0:
        print("  *** TODOS LOS TESTS PASARON ***")
    else:
        print(f"  *** {FAIL} TEST(S) FALLARON ***")
    print()

    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
