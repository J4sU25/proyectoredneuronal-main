#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import runpy
import linecache

print("Iniciando programa maestro...\n")

# Paso 1: Ejecutar entrenamiento de epocas
print("="*100)
print("  ENTRENAMIENTO DE EPOCAS (RED NEURONAL)")
print("="*100 + "\n")

try:
    ruta1 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proyecto ADA epocas', 'proyecto final ADA 1.0'))
    os.chdir(ruta1)
    if ruta1 not in sys.path:
        sys.path.insert(0, ruta1)
    
    # Importar y ejecutar main
    from main import main
    main()
    
except Exception as e:
    print(f"\nError en paso 1: {e}")
    import traceback
    traceback.print_exc()

# Paso 2: Ejecutar analisis de codigo  
print("\n" + "="*100)
print("  ANALISIS DE CODIGO (RED NEURONAL)")
print("="*100 + "\n")

try:
    ruta2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proyecto ADA analiz', 'proyecto ADA anal'))
    os.chdir(ruta2)
    
    if ruta2 not in sys.path:
        sys.path.insert(0, ruta2)
    
    # Leer el archivo mi_codigo.py
    with open('mi_codigo.py', 'r', encoding='utf-8') as f:
        codigo_original = f.read()
    
    # Usar linecache para que inspect.getsource() funcione
    nombre_archivo = os.path.abspath('mi_codigo.py')
    linecache.cache[nombre_archivo] = (
        len(codigo_original),
        None,
        [linea + '\n' for linea in codigo_original.split('\n')],
        nombre_archivo
    )
    
    # Crear namespace con variables globales necesarias
    namespace = {
        '__name__': '__main__',
        '__file__': nombre_archivo,
        '__builtins__': __builtins__,
        'sys': sys,
        'os': os
    }
    
    # Compilar y ejecutar en el namespace
    codigo_compilado = compile(codigo_original, nombre_archivo, 'exec')
    exec(codigo_compilado, namespace)
    
except Exception as e:
    print(f"\nError en paso 2: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*100)
print("[COMPLETADO]")
print("="*100)
