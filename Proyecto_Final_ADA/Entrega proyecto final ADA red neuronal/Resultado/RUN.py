#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script maestro de ejecucion - Proyecto Final ADA
Orquesta: Pruebas unitarias -> Entrenamiento completo (RA1-RA4) -> Analisis de codigo
"""

import sys
import os
import traceback

print("=" * 100)
print("  PROYECTO FINAL ADA - EJECUCION COMPLETA")
print("  Red Neuronal MLP desde cero | Analisis de Complejidad | Estructuras de Datos")
print("=" * 100 + "\n")

# ============================================================
# PASO 0: PRUEBAS UNITARIAS
# ============================================================
print("=" * 100)
print("  PASO 0: PRUEBAS UNITARIAS")
print("=" * 100 + "\n")

try:
    ruta_pruebas = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Pruebas')
    )
    os.chdir(ruta_pruebas)
    if ruta_pruebas not in sys.path:
        sys.path.insert(0, ruta_pruebas)

    # Ejecutar pruebas
    import subprocess
    resultado_tests = subprocess.run(
        [sys.executable, 'test_mlp.py'],
        capture_output=False,
        text=True
    )
    tests_ok = resultado_tests.returncode == 0

    if tests_ok:
        print("\n  [OK] Todas las pruebas pasaron. Continuando con el entrenamiento...\n")
    else:
        print("\n  [!] Algunas pruebas fallaron. Continuando de todas formas...\n")

except Exception as e:
    print(f"\n  Error ejecutando pruebas: {e}")
    traceback.print_exc()
    tests_ok = False

# ============================================================
# PASO 1: ENTRENAMIENTO COMPLETO (RA1 - RA4)
# ============================================================
print("\n" + "=" * 100)
print("  PASO 1: ENTRENAMIENTO Y ANALISIS COMPLETO (FASES RA1 - RA4)")
print("=" * 100 + "\n")

try:
    ruta_entrenamiento = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'Epocas', 'Entrenamiento_ADA 1.0')
    )
    os.chdir(ruta_entrenamiento)
    if ruta_entrenamiento not in sys.path:
        sys.path.insert(0, ruta_entrenamiento)

    from main import main as main_entrenamiento
    main_entrenamiento()

except Exception as e:
    print(f"\n  Error en el entrenamiento: {e}")
    traceback.print_exc()

# ============================================================
# PASO 2: ANALISIS DE CODIGO EXTERNO
# ============================================================
print("\n" + "=" * 100)
print("  PASO 2: ANALISIS DE ALGORITMO EXTERNO (mi_codigo.py)")
print("=" * 100 + "\n")

try:
    import linecache

    ruta_analisis = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'Analisis', 'Proyecto_Analisis')
    )
    os.chdir(ruta_analisis)
    if ruta_analisis not in sys.path:
        sys.path.insert(0, ruta_analisis)

    with open('mi_codigo.py', 'r', encoding='utf-8') as f:
        codigo_original = f.read()

    nombre_archivo = os.path.abspath('mi_codigo.py')
    linecache.cache[nombre_archivo] = (
        len(codigo_original),
        None,
        [linea + '\n' for linea in codigo_original.split('\n')],
        nombre_archivo
    )

    namespace = {
        '__name__': '__main__',
        '__file__': nombre_archivo,
        '__builtins__': __builtins__,
        'sys': sys,
        'os': os
    }

    codigo_compilado = compile(codigo_original, nombre_archivo, 'exec')
    exec(codigo_compilado, namespace)

except Exception as e:
    print(f"\n  Error en analisis de codigo: {e}")
    traceback.print_exc()

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 100)
print("  EJECUCION COMPLETA FINALIZADA")
print("=" * 100)
print(f"\n  Pruebas unitarias: {'PASADAS' if tests_ok else 'CON FALLOS'}")
print(f"  Fases RA1-RA4:     Ejecutadas")
print(f"  Analisis externo:  Ejecutado")
print("\n  Archivos generados en model_state/:")
print("    - weights_checkpoint.json  (pesos del modelo)")
print("    - training_history.json    (historial de entrenamiento)")
print("    - complexity_analysis.json (analisis de complejidad)")
print("\n" + "=" * 100 + "\n")
