"""
Verificacion Numerica de Gradientes (Gradient Check)
Verifica que el backpropagation calcula gradientes correctos

Criterio de correctitud del enunciado: error relativo < 1e-4

Metodo: diferencias finitas centradas
    grad_numerico(f, theta, i) = (f(theta + eps*e_i) - f(theta - eps*e_i)) / (2*eps)
"""

import math
import random
from typing import List, Tuple, Dict
from src.matrix import Matrix


def _perturb_weight(model, layer_idx: int, i: int, j: int, delta: float) -> None:
    """Perturba el peso W[i][j] de la capa layer_idx en delta."""
    model.layers[layer_idx].W.data[i][j] += delta


def _perturb_bias(model, layer_idx: int, j: int, delta: float) -> None:
    """Perturba el sesgo b[0][j] de la capa layer_idx en delta."""
    model.layers[layer_idx].b.data[0][j] += delta


def _compute_loss(model, X: Matrix, y: Matrix) -> float:
    """Calcula la perdida total del modelo con los datos dados."""
    from src.activation import softmax
    import math

    predictions = model.forward(X)
    loss = model.compute_loss(predictions, y)
    return loss


def gradient_check(model,
                   X: Matrix,
                   y: Matrix,
                   epsilon: float = 1e-5,
                   num_checks: int = 20,
                   verbose: bool = True) -> Dict[str, float]:
    """
    Verificacion numerica de gradientes por diferencias finitas centradas.

    Para cada parametro theta_i seleccionado aleatoriamente:
        grad_numerico = (L(theta + eps) - L(theta - eps)) / (2 * eps)
        grad_analitico = dL/d_theta_i  (calculado por backprop)

        error_relativo = |grad_numerico - grad_analitico| /
                         max(|grad_numerico|, |grad_analitico|, 1e-8)

    El error relativo debe ser < 1e-4 para considerar el gradiente correcto.

    Complejidad temporal: O(num_checks * (forward_cost + backward_cost))
    Complejidad espacial: O(total_params)

    Parametros:
    - model: instancia del MLP
    - X: datos de entrada (batch pequeno)
    - y: etiquetas verdaderas
    - epsilon: tamano de la perturbacion
    - num_checks: numero de pesos a verificar aleatoriamente
    - verbose: si True, imprime detalles

    Retorna: diccionario con estadisticas del gradient check
    """

    if verbose:
        print("\n" + "=" * 70)
        print("  VERIFICACION NUMERICA DE GRADIENTES (Gradient Check)")
        print("=" * 70)
        print(f"\n  Metodo: Diferencias finitas centradas")
        print(f"  epsilon = {epsilon}")
        print(f"  Criterio de correctitud: error relativo < 1e-4")
        print(f"  Muestras verificadas: {num_checks} pesos aleatorios")
        print()

    # -------------------------------------------------------
    # Paso 1: Calcular gradientes analiticos con backprop
    # -------------------------------------------------------
    # Forward pass
    from src.activation import softmax_derivative
    predictions = model.forward(X)
    loss_base = model.compute_loss(predictions, y)

    # Backward pass (sin actualizar pesos, solo calcular gradientes)
    # Guardamos los gradientes antes de modificar pesos
    dLoss = softmax_derivative(predictions, y)

    # Simular backward sin actualizar pesos para obtener gradientes
    grad_analyticals = {}
    dA = dLoss

    # Propagar hacia atras y recolectar gradientes
    for layer_idx in reversed(range(len(model.layers))):
        layer = model.layers[layer_idx]
        batch_size = dA.rows

        # Derivada de activacion
        from src.activation import ActivationFunction
        dZ = ActivationFunction.apply_derivative(layer.output_cache, layer.activation)
        dZ = dZ.element_wise_mult(dA)

        # Gradiente de pesos
        X_T = layer.input_cache.transpose()
        dW = X_T.dot(dZ).scalar_mult(1.0 / batch_size)

        # Gradiente de sesgo
        db = Matrix.zeros(1, layer.output_size)
        for j in range(layer.output_size):
            s = 0.0
            for bi in range(batch_size):
                s += dZ.data[bi][j]
            db.data[0][j] = s / batch_size

        # Gradiente para capa anterior
        W_T = layer.W.transpose()
        dA = dZ.dot(W_T)

        grad_analyticals[layer_idx] = {
            'dW': [row[:] for row in dW.data],
            'db': [row[:] for row in db.data]
        }

    # -------------------------------------------------------
    # Paso 2: Calcular gradientes numericos y comparar
    # -------------------------------------------------------
    errors = []
    check_results = []

    # Colectar todos los parametros disponibles
    all_params = []
    for layer_idx, layer in enumerate(model.layers):
        for i in range(layer.input_size):
            for j in range(layer.output_size):
                all_params.append(('W', layer_idx, i, j))
        for j in range(layer.output_size):
            all_params.append(('b', layer_idx, 0, j))

    # Seleccionar aleatoriamente num_checks parametros
    if num_checks > len(all_params):
        num_checks = len(all_params)

    random.seed(42)
    selected = random.sample(all_params, num_checks)

    for param_type, layer_idx, i, j in selected:
        layer = model.layers[layer_idx]

        # Guardar valor original
        if param_type == 'W':
            original = layer.W.data[i][j]
            grad_analitico = grad_analyticals[layer_idx]['dW'][i][j]
        else:
            original = layer.b.data[i][j]
            grad_analitico = grad_analyticals[layer_idx]['db'][i][j]

        # Perturbacion positiva: f(theta + eps)
        if param_type == 'W':
            layer.W.data[i][j] = original + epsilon
        else:
            layer.b.data[i][j] = original + epsilon

        loss_plus = _compute_loss(model, X, y)

        # Perturbacion negativa: f(theta - eps)
        if param_type == 'W':
            layer.W.data[i][j] = original - epsilon
        else:
            layer.b.data[i][j] = original - epsilon

        loss_minus = _compute_loss(model, X, y)

        # Restaurar valor original
        if param_type == 'W':
            layer.W.data[i][j] = original
        else:
            layer.b.data[i][j] = original

        # Gradiente numerico
        grad_numerico = (loss_plus - loss_minus) / (2 * epsilon)

        # Error relativo
        numerador = abs(grad_numerico - grad_analitico)
        denominador = max(abs(grad_numerico), abs(grad_analitico), 1e-8)
        error_relativo = numerador / denominador

        errors.append(error_relativo)

        estado = "OK" if error_relativo < 1e-4 else "FALLO"
        check_results.append({
            'param': f"{param_type}[{layer_idx}][{i},{j}]",
            'analitico': grad_analitico,
            'numerico': grad_numerico,
            'error': error_relativo,
            'estado': estado
        })

    # -------------------------------------------------------
    # Paso 3: Reporte
    # -------------------------------------------------------
    max_error = max(errors) if errors else 0
    avg_error = sum(errors) / len(errors) if errors else 0
    num_ok = sum(1 for r in check_results if r['estado'] == 'OK')
    num_fail = num_checks - num_ok

    if verbose:
        print(f"  {'Parametro':<25} | {'Analitico':>14} | {'Numerico':>14} | {'Error rel.':>12} | Estado")
        print("  " + "-" * 85)
        for r in check_results[:10]:  # Mostrar primeros 10
            print(f"  {r['param']:<25} | {r['analitico']:>14.8f} | {r['numerico']:>14.8f} | "
                  f"{r['error']:>12.2e} | {r['estado']}")

        if num_checks > 10:
            print(f"  ... ({num_checks - 10} mas no mostrados)")

        print("\n  " + "-" * 85)
        print(f"\n  Resumen:")
        print(f"    Pesos verificados: {num_checks}")
        print(f"    Correctos (error < 1e-4): {num_ok} / {num_checks}")
        print(f"    Fallidos: {num_fail}")
        print(f"    Error maximo: {max_error:.2e}")
        print(f"    Error promedio: {avg_error:.2e}")

        if max_error < 1e-4:
            print(f"\n  [OK] BACKPROPAGATION CORRECTO")
            print(f"       El error maximo ({max_error:.2e}) esta por debajo del umbral 1e-4")
        elif max_error < 1e-2:
            print(f"\n  [ADVERTENCIA] Error moderado: {max_error:.2e}")
            print(f"       Posible problema numerico, verificar epsilon y arquitectura")
        else:
            print(f"\n  [ERROR] Error alto en gradientes: {max_error:.2e}")
            print(f"       Backpropagation puede tener un bug")

    return {
        'max_error': max_error,
        'avg_error': avg_error,
        'num_ok': num_ok,
        'num_fail': num_fail,
        'num_checks': num_checks,
        'passed': max_error < 1e-4,
        'details': check_results
    }
