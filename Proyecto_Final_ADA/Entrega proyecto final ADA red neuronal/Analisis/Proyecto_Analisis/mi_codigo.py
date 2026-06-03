#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALISIS DE ALGORITMOS
========================
Reemplace el código dentro de def mi_algoritmo(lista):
Luego ejecute: python mi_codigo.py
"""

def mi_algoritmo(arr):
    """Reemplace este código con el suyo"""
    def heapify(arr, n, i):
        mayor = i
        izq = 2 * i + 1
        der = 2 * i + 2

        if izq < n and arr[izq] > arr[mayor]:
            mayor = izq
        if der < n and arr[der] > arr[mayor]:
            mayor = der

        if mayor != i:
            arr[i], arr[mayor] = arr[mayor], arr[i]
            heapify(arr, n, mayor)

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr



if __name__ == "__main__":
    from analizar import AnalizadorAlgoritmos
    import inspect
    import time
    import tracemalloc
    from datetime import datetime

    # Analizar
    analizador = AnalizadorAlgoritmos()
    analizador.entrenar_red_neuronal()

    codigo = inspect.getsource(mi_algoritmo)
    features = analizador._analizar_codigo(codigo)
    complejidad = analizador._estimar_big_o(features)
    big_o = complejidad['big_o']
    big_omega = complejidad['big_omega']
    big_theta = complejidad['big_theta']

    # Medir tiempo y memoria
    entrada_prueba = [1, 2, 3, 2, 4, 5, 3, 6, 7, 2, 8, 9, 1, 5, 4, 7, 3, 1]
    tracemalloc.start()
    inicio = time.perf_counter()
    resultado = mi_algoritmo(entrada_prueba)
    fin = time.perf_counter()
    memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tiempo_real = (fin - inicio) * 1000
    memoria_real = memoria_pico / 1024
    tiempo_pred, memoria_pred = analizador.predecir_complejidad(features)

    # Generar reporte
    reporte = f"""
{'='*80}
ANALISIS COMPLETO DEL ALGORITMO
{'='*80}

NOMBRE: {mi_algoritmo.__name__}
FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
BIG-O (COMPLEJIDAD TEORICA)
{'='*80}

Big-O (Cota Superior): {big_o}
Big-Omega (Cota Inferior): {big_omega}
Big-Theta (Cota Ajustada): {big_theta}

Analisis de Codigo:
  Loops detectados: {features['loops_simples']}
  Loops anidados: {features['loops_anidados']}
  Recursion: {'Si' if features['recursion'] else 'No'}
  Bifurcaciones: {features['bifurcaciones']}
  Lineas de codigo: {features['longitud_lineas']}

{'='*80}
TIEMPO REAL (MEDICION PRACTICA)
{'='*80}

Tiempo de Ejecucion: {tiempo_real:.6f} ms
Datos Usados: {len(entrada_prueba)} elementos
Resultado: {resultado}

{'='*80}
MEMORIA (CONSUMO REAL)
{'='*80}

Memoria Utilizada: {memoria_real:.2f} KB
Memoria Pico: {memoria_pico / 1024:.2f} KB

{'='*80}
ANALISIS RED NEURONAL (PREDICCION IA)
{'='*80}

Tiempo Predicho: {tiempo_pred:.6f} ms
Memoria Predicha: {memoria_pred:.2f} KB

Comparacion Teoria vs Practica:
  Big-O: {big_o}
  Tiempo Real: {tiempo_real:.6f} ms
  Tiempo Prediccion: {tiempo_pred:.6f} ms
  Memoria Real: {memoria_real:.2f} KB
  Memoria Prediccion: {memoria_pred:.2f} KB

{'='*80}
VERIFICACION (CORRECTITUD)
{'='*80}

Funcion ejecutada exitosamente
Resultado: {resultado}

{'='*80}
REPORTE FINAL
{'='*80}

Algoritmo: {mi_algoritmo.__name__}
Big-O: {big_o}
Big-Omega: {big_omega}
Big-Theta: {big_theta}
Tiempo Real: {tiempo_real:.6f} ms
Memoria: {memoria_real:.2f} KB
Prediccion NN - Tiempo: {tiempo_pred:.6f}ms
Prediccion NN - Memoria: {memoria_pred:.2f}KB

{'='*80}
"""

    print(reporte)
