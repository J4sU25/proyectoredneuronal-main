import urllib.request
import urllib.parse
import json
import time
import os
import traceback
import sys
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------
# CONFIGURACION DEL BOT
# ---------------------------------------------------------

TOKEN = "8810286121:AAEi7DhuEI59fsc71OIIlR8fRlO9l0hb_GA"

URL_API = "https://api.telegram.org/bot"

# ---------------------------------------------------------
# INTEGRACION CON LA RED NEURONAL (ADA) Y ALGORITMOS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.mlp import MLP
    from src.matrix import Matrix
    from src.dataset import DatasetGenerator
    from src.knn import KNNClassifier
    from src.sorting import heapsort, quicksort
    from src.data_structures import quickselect_top_k, BST
    from Analisis.Proyecto_Analisis.analizar import AnalizadorAlgoritmos
except ImportError as e:
    print(f"Error importando modulos del proyecto: {e}")
    sys.exit(1)

# Variables globales para el modelo
MODEL_CHECKPOINT = os.path.join(BASE_DIR, 'model_state', 'weights_checkpoint.json')
HISTORY_PATH = os.path.join(BASE_DIR, 'model_state', 'training_history.json')

# Instancia global del modelo (se inicializa al primer uso)
mlp_bot = None
dataset_cache = None # Para entrenamiento rapido (features=20, classes=10)

def inicializar_modelo():
    global mlp_bot
    if mlp_bot is None:
        mlp_bot = MLP([20, 4, 10], ['relu', 'linear'], 'cross_entropy')
        if os.path.exists(MODEL_CHECKPOINT):
            mlp_bot.load_weights(MODEL_CHECKPOINT)
            print("Memoria del MLP cargada correctamente.")
        else:
            print("No se encontro memoria previa. Se usara un modelo inicializado aleatoriamente.")

def obtener_datos_entrenamiento():
    global dataset_cache
    if dataset_cache is None:
        # Generar datos consistentes con main.py
        X, y = DatasetGenerator.generate_synthetic_classification(
            num_samples=800, num_features=20, num_classes=10, seed=42
        )
        X = DatasetGenerator.normalize_dataset(X)
        dataset_cache = (X, y)
    return dataset_cache

# ---------------------------------------------------------
# FUNCIONES DE LA API DE TELEGRAM
# ---------------------------------------------------------

def hacer_peticion(metodo: str, parametros: dict = None) -> dict:
    """Realiza una peticion HTTP a la API de Telegram."""
    if not TOKEN:
        raise ValueError("EL TOKEN DEL BOT NO ESTA CONFIGURADO.")
        
    url = f"{URL_API}{TOKEN}/{metodo}"
    
    if parametros:
        # Convertir parametros a datos urlencoded
        data = urllib.parse.urlencode(parametros).encode('utf-8')
        req = urllib.request.Request(url, data=data)
    else:
        req = urllib.request.Request(url)
        
    try:
        with urllib.request.urlopen(req) as response:
            respuesta_json = response.read().decode('utf-8')
            return json.loads(respuesta_json)
    except Exception as e:
        print(f"Error de conexion con Telegram: {e}")
        return None

def enviar_mensaje(chat_id: int, texto: str):
    """Envia un mensaje de texto al chat especificado."""
    hacer_peticion("sendMessage", {"chat_id": chat_id, "text": texto})

# ---------------------------------------------------------
# PROCESAMIENTO DE COMANDOS
# ---------------------------------------------------------

# Diccionario para rastrear chats que estan esperando codigo
esperando_codigo = {}  # {chat_id: True}

def analizar_codigo_ast(codigo: str) -> dict:
    """Analiza codigo Python con AST y retorna complejidad estimada."""
    import ast as ast_mod
    try:
        arbol = ast_mod.parse(codigo)
    except SyntaxError as e:
        return {'error': f'Error de sintaxis en el codigo: {e}'}

    loops = 0
    max_depth = 0
    recursion = False
    funciones = set()
    llamadas = set()

    class Visitor(ast_mod.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.max_d = 0
        def visit_FunctionDef(self, node):
            funciones.add(node.name)
            self.generic_visit(node)
        def visit_For(self, node):
            nonlocal loops, max_depth
            loops += 1
            self.depth += 1
            if self.depth > self.max_d:
                self.max_d = self.depth
                max_depth = self.max_d
            self.generic_visit(node)
            self.depth -= 1
        def visit_While(self, node):
            nonlocal loops, max_depth
            loops += 1
            self.depth += 1
            if self.depth > self.max_d:
                self.max_d = self.depth
                max_depth = self.max_d
            self.generic_visit(node)
            self.depth -= 1
        def visit_Call(self, node):
            if isinstance(node.func, ast_mod.Name):
                llamadas.add(node.func.id)
            self.generic_visit(node)

    v = Visitor()
    v.visit(arbol)

    # Detectar recursion
    for f in funciones:
        if f in llamadas:
            recursion = True
            break

    lineas = len([l for l in codigo.split('\n') if l.strip()])

    # Determinar Big-O
    if recursion and loops > 0:
        big_o, big_omega, big_theta = 'O(n log n)', 'Omega(n log n)', 'Theta(n log n)'
    elif max_depth >= 3:
        big_o, big_omega, big_theta = f'O(n^{max_depth})', f'Omega(n^{max_depth-1})', f'Theta(n^{max_depth})'
    elif max_depth == 2:
        big_o, big_omega, big_theta = 'O(n^2)', 'Omega(n)', 'Theta(n^2)'
    elif recursion:
        big_o, big_omega, big_theta = 'O(2^n)', 'Omega(n)', 'Theta(2^n)'
    elif loops == 1:
        big_o, big_omega, big_theta = 'O(n)', 'Omega(1)', 'Theta(n)'
    elif loops > 1:
        big_o, big_omega, big_theta = 'O(n)', 'Omega(1)', 'Theta(n)'
    else:
        big_o, big_omega, big_theta = 'O(1)', 'Omega(1)', 'Theta(1)'

    return {
        'big_o': big_o, 'big_omega': big_omega, 'big_theta': big_theta,
        'loops': loops, 'max_depth': max_depth,
        'recursion': recursion, 'lineas': lineas,
        'funciones': list(funciones)
    }

def manejar_comando(chat_id: int, texto: str):
    global esperando_codigo
    print(f"Comando recibido: {texto}")
    texto_strip = texto.strip()

    # Si el chat esta esperando codigo para analizar
    if chat_id in esperando_codigo:
        del esperando_codigo[chat_id]
        if texto_strip.startswith('/'):
            enviar_mensaje(chat_id, "Analisis cancelado. Escribe /complejidad para intentar de nuevo.")
            return
        # Analizar el codigo recibido
        resultado = analizar_codigo_ast(texto_strip)
        if 'error' in resultado:
            enviar_mensaje(chat_id, f"Error al analizar: {resultado['error']}")
            return
        msj = (
            f"Analisis de Complejidad (AST)\n\n"
            f"Big-O (Peor Caso):    {resultado['big_o']}\n"
            f"Big-Omega (Mejor):    {resultado['big_omega']}\n"
            f"Big-Theta (Ajuste):   {resultado['big_theta']}\n\n"
            f"Detalles:\n"
            f"  Loops encontrados:  {resultado['loops']}\n"
            f"  Max anidamiento:    {resultado['max_depth']}\n"
            f"  Recursion:          {'Si' if resultado['recursion'] else 'No'}\n"
            f"  Lineas de codigo:   {resultado['lineas']}\n"
            f"  Funciones:          {', '.join(resultado['funciones']) or 'ninguna'}\n"
        )
        enviar_mensaje(chat_id, msj)
        return

    texto = texto_strip
    
    if texto.startswith('/start') or texto.startswith('/help'):
        ayuda = (
            "🤖 Bot Red Neuronal ADA\n\n"
            "Soy un bot conectado a tu red neuronal y Proyecto de Algoritmos.\n\n"
            "Comandos disponibles:\n"
            "/status - Ver estadisticas de la red neuronal\n"
            "/entrenar <numero de epocas> - Entrena la red en el numero de epocas que desees\n"
            "/predecir - Predice la clase pasando 20 numeros\n"
            "/clasificar_digito - MLP vs k-NN con datos aleatorios\n"
            "/ordenamiento - Heapsort vs Quicksort\n"
            "/estructuras - Demostracion Quickselect Top-K y BST\n"
            "/complejidad - Analisis estatico Big-O de tu codigo\n"
        )
        enviar_mensaje(chat_id, ayuda)
        
    elif texto.startswith('/status'):
        if not os.path.exists(HISTORY_PATH):
            enviar_mensaje(chat_id, "El modelo aun no ha sido entrenado.")
            return
            
        with open(HISTORY_PATH, 'r') as f:
            historia = json.load(f)
            
        total_epocas = historia.get('total_epochs_trained', historia.get('total_epochs', 0))
        total_ejecuciones = historia.get('total_executions', 0)
        ultima_perdida = historia['all_losses'][-1] if historia.get('all_losses') else historia.get('final_loss', 0)
        ultima_precision = historia['all_accuracies'][-1] if historia.get('all_accuracies') else historia.get('final_accuracy', 0)
        
        estado = (
            f"📊 *Estado de la Red Neuronal*\n"
            f"• Total de ejecuciones: {total_ejecuciones}\n"
            f"• Epocas totales entrenadas: {total_epocas}\n"
            f"• Ultima perdida: {ultima_perdida:.4f}\n"
            f"• Ultima precision: {ultima_precision*100:.2f}%\n"
            f"• Arquitectura: 1 Capa Oculta (4 Neuronas)\n"
        )
        enviar_mensaje(chat_id, estado)
        
    elif texto.startswith('/entrenar'):
        # Extraer el numero de epocas
        partes = texto.split()
        epocas = 10
        if len(partes) > 1 and partes[1].isdigit():
            epocas = int(partes[1])
            
        if epocas > 2000:
            enviar_mensaje(chat_id, "Para evitar demoras excesivas, entrena un maximo de 2000 epocas por comando.")
            return
            
        enviar_mensaje(chat_id, f"Entrenando la red por {epocas} epocas...\n(Esto puede tomar tiempo, por favor espera)")
        
        inicializar_modelo()
        X, y = obtener_datos_entrenamiento()
        
        # Guardar loss inicial
        perdida_inicial, _ = mlp_bot.train_epoch(X, y, batch_size=32, learning_rate=0.01)
        
        for _ in range(epocas - 1):
            perdida_final, acc_final = mlp_bot.train_epoch(X, y, batch_size=32, learning_rate=0.01)
            
        # Guardar pesos
        mlp_bot.save_weights(MODEL_CHECKPOINT)
        
        # Actualizar historial en el mismo formato que main.py
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                historia = json.load(f)
        else:
            historia = {}

        # Asegurar que todos los campos existen (compatibilidad con versiones anteriores)
        historia.setdefault('total_executions', 0)
        historia.setdefault('total_epochs_trained', historia.get('total_epochs', 0))
        historia.setdefault('all_losses', [])
        historia.setdefault('all_accuracies', [])
        historia.setdefault('execution_history', [])

        historia['total_executions']   += 1
        historia['total_epochs_trained'] += epocas
        total_epochs = historia['total_epochs_trained']
        historia['all_losses'].append(perdida_final)
        historia['all_accuracies'].append(acc_final)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(historia, f, indent=2)

            
        enviar_mensaje(chat_id, 
            f"✅ *Entrenamiento finalizado*\n\n"
            f"📈 Precision actual: {acc_final*100:.2f}%\n"
            f"📉 Perdida final: {perdida_final:.4f}\n"
            f"⏳ *Total acumulado de epocas:* {total_epochs}\n\n"
            f"Memoria de la red guardada correctamente en disco."
        )

    elif texto.startswith('/predecir'):
        argumentos = texto[len('/predecir'):].strip()
        if not argumentos:
            enviar_mensaje(chat_id, "Por favor, ingresa los datos numericos separados por coma. Ejemplo: /predecir 0.1, -0.5, 1.2 ... (hasta 20 datos)")
            return
            
        valores_str = argumentos.split(',')
        try:
            valores = [float(x.strip()) for x in valores_str]
        except ValueError:
            enviar_mensaje(chat_id, "Error: Todos los valores deben ser numericos.")
            return
            
        if len(valores) != 20:
            # Rellenar con ceros o cortar para asegurar que sean 20 caracteristicas
            if len(valores) < 20:
                valores.extend([0.0] * (20 - len(valores)))
            else:
                valores = valores[:20]
                
        inicializar_modelo()
        
        # Crear matriz 1x20
        X_test = Matrix(1, 20, [valores])
        
        # Hacer Forward Pass
        pred_probs = mlp_bot.forward(X_test)
        
        # Obtener la clase con mayor probabilidad (Softmax)
        probabilidades = pred_probs.data[0]
        clase_predicha = probabilidades.index(max(probabilidades))
        confianza = max(probabilidades) * 100
        
        enviar_mensaje(chat_id, 
            f"🧠 *Resultado de la Prediccion*\n"
            f"• Clase predicha: {clase_predicha} (de 10)\n"
            f"• Nivel de confianza: {confianza:.2f}%\n\n"
            f"(Nota: La red analizo el vector a traves de su capa oculta de 4 neuronas para este resultado)."
        )

    elif texto.startswith('/clasificar_digito'):
        import random
        enviar_mensaje(chat_id, "Generando digito aleatorio y ejecutando modelos...")
        # Simular una imagen 8x8 aplanada (64 features), simplificado a 20 features para usar el MLP instanciado
        X, y = DatasetGenerator.generate_mnist_like(num_samples=10, num_classes=5, img_size=5, seed=random.randint(1,1000))
        # MLP de 25 features (5x5). Instanciamos un MLP especifico para la prueba
        mlp_temp = MLP([25, 4, 5], ['relu', 'linear'], 'cross_entropy')
        mlp_temp.train_epoch(X, y, 10, 0.1) # Entrenamiento rapido minimo
        
        muestra_idx = random.randint(0, 9)
        muestra = Matrix(1, 25, [X.data[muestra_idx]])
        target_clase = y.data[muestra_idx].index(1)
        
        # Prediccion MLP
        pred_mlp = mlp_temp.forward(muestra).data[0]
        clase_mlp = pred_mlp.index(max(pred_mlp))
        
        # Prediccion KNN (k=1)
        knn = KNNClassifier(k=1)
        knn.fit(X, y)
        pred_knn = knn.predict(muestra)
        clase_knn = pred_knn[0]
        
        msj = (
            f"🖼️ *Carrera de Clasificacion de Digitos*\n"
            f"• Clase real del digito: {target_clase}\n\n"
            f"🤖 *MLP (Red Neuronal)*: {clase_mlp}\n"
            f"📐 *k-NN (Vecinos)*: {clase_knn}\n\n"
            f"(Generalmente k-NN = 1 es infalible si la muestra esta en los datos de entrenamiento)"
        )
        enviar_mensaje(chat_id, msj)

    elif texto.startswith('/ordenamiento'):
        import random
        import time
        enviar_mensaje(chat_id, "Ordenando 1,000 elementos aleatorios... 🏃‍♂️")
        datos = [random.uniform(0, 1000) for _ in range(1000)]
        
        # Heapsort
        t0 = time.perf_counter()
        heapsort(datos)
        t_heap = (time.perf_counter() - t0) * 1000
        
        # Quicksort
        t0 = time.perf_counter()
        quicksort(datos)
        t_quick = (time.perf_counter() - t0) * 1000
        
        msj = (
            f"⏱️ *Carrera de Ordenamiento (1000 elementos)*\n"
            f"• Heapsort (O(n log n)): {t_heap:.4f} ms\n"
            f"• Quicksort (promedio O(n log n)): {t_quick:.4f} ms\n\n"
            f"Quicksort suele ser un poco mas rapido en memoria cache, "
            f"pero Heapsort garantiza O(n log n) en el peor de los casos."
        )
        enviar_mensaje(chat_id, msj)

    elif texto.startswith('/estructuras'):
        enviar_mensaje(chat_id, "Analizando estructuras de datos... 🏗️")
        
        # Quickselect
        import random
        perdidas = [random.uniform(0.1, 2.0) for _ in range(500)]
        top_k_indices = quickselect_top_k(perdidas, k=3)
        top_k_vals = sorted([perdidas[i] for i in top_k_indices], reverse=True)
        
        # BST
        bst = BST()
        bst.insert(0.5, 0, 0, 0)
        bst.insert(0.1, 0, 0, 1)
        bst.insert(0.8, 0, 1, 0)
        
        msj = (
            f"⚙️ *Estructuras de Datos*\n\n"
            f"📌 *Quickselect (Hard Mining)*\n"
            f"Se buscaron las 3 peores perdidas de 500 muestras en tiempo O(n):\n"
            f"Top 3 perdidas: {top_k_vals[0]:.2f}, {top_k_vals[1]:.2f}, {top_k_vals[2]:.2f}\n\n"
            f"🌲 *Árbol Binario (BST) para Poda*\n"
            f"Se instancio un arbol y se inserto la memoria de 3 conexiones.\n"
            f"• Menor valor de memoria en O(log n): {bst.find_min().magnitude:.2f}\n"
            f"• Mayor valor de memoria en O(log n): {bst.find_max().magnitude:.2f}\n"
        )
        enviar_mensaje(chat_id, msj)

    elif texto.startswith('/complejidad'):
        esperando_codigo[chat_id] = True
        enviar_mensaje(chat_id,
            "Analizador de Complejidad (AST)\n\n"
            "Pega tu codigo Python en el siguiente mensaje y lo analizare con el Arbol de Sintaxis Abstracta (AST).\n\n"
            "Detectare: loops, anidamiento, recursion y te dare el Big-O, Big-Omega y Big-Theta.\n\n"
            "(Si quieres cancelar, escribe /cancelar)"
        )

    else:
        enviar_mensaje(chat_id, "Comando no reconocido. Escribe /help para ver las opciones.")

# ---------------------------------------------------------
# CICLO PRINCIPAL DEL BOT (LONG POLLING)
# ---------------------------------------------------------

def iniciar_bot():
    if not TOKEN:
        print("ERROR")
        return
        
    print("Conectando con Telegram...")
    # Verificar conexion
    me = hacer_peticion("getMe")
    if not me or not me.get("ok"):
        print("Fallo al conectar con Telegram. Revisa el TOKEN y tu conexion a internet.")
        return
        
    print(f"Conectado exitosamente como: @{me['result']['username']}")
    print("Bot en escucha de mensajes (presiona Ctrl+C para detener)...")
    
    offset = 0
    
    while True:
        try:
            # Usar timeout largo para Long Polling
            updates = hacer_peticion("getUpdates", {"offset": offset, "timeout": 30})
            
            if updates and updates.get("ok") and updates.get("result"):
                for update in updates["result"]:
                    # Actualizar el offset para no volver a procesar el mismo mensaje
                    offset = update["update_id"] + 1
                    
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        texto = update["message"]["text"]
                        
                        try:
                            manejar_comando(chat_id, texto)
                        except Exception as e:
                            print(f"Error procesando comando: {e}")
                            traceback.print_exc()
                            enviar_mensaje(chat_id, f"Hubo un error interno al procesar el comando.")
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nDeteniendo el bot...")
            break
        except Exception as e:
            print(f"Error en el ciclo principal: {e}")
            time.sleep(5)

if __name__ == "__main__":
    iniciar_bot()
