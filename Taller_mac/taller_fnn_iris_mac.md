# Taller: Feedforward Neural Network con el dataset Iris — macOS

En este taller construirás, entrenarás y evaluarás una red neuronal feedforward (FNN) que clasifica flores Iris en 3 especies usando **Python + TensorFlow/Keras**.

**Duración estimada:** 45–60 minutos (la mayor parte es la instalación).

## 1. ¿Qué vamos a hacer?

| Elemento | Detalle |
|---|---|
| Dataset | Iris: 150 flores, 4 medidas (largo/ancho de sépalo y pétalo) |
| Objetivo | Clasificar en *setosa*, *versicolor* o *virginica* |
| Red | 4 entradas → 16 neuronas (ReLU) → 8 neuronas (ReLU) → 3 salidas (softmax) |
| Pérdida | Sparse categorical crossentropy |
| Optimizador | Adam (variante de descenso por gradiente) |

## 2. Requisitos

- macOS 12 (Monterey) o superior.
- Mac con **Apple Silicon** (M1/M2/M3/M4) o **Intel** (ver notas específicas en el paso 4).
- ~2 GB de espacio libre en disco.
- Conexión a internet.
- **Herramientas de línea de comandos de Xcode** y **Homebrew** (se instalan en el paso 3).
- No necesitas GPU: el taller corre con CPU.

**Librerías de Python** (se instalan en el paso 4): `tensorflow`, `scikit-learn`, `matplotlib` (numpy se instala como dependencia).

> ⚠️ **Versión de Python:** usa **3.10, 3.11 o 3.12**. Las versiones muy nuevas (3.13+) pueden no tener paquetes de TensorFlow disponibles todavía. Verifica la compatibilidad actual en la [guía oficial de instalación de TensorFlow](https://www.tensorflow.org/install/pip).

## 3. Instalación de Python y preparación del entorno

Abre la app **Terminal** (Spotlight → “Terminal”).

**3.1 Herramientas de Xcode:**

```bash
xcode-select --install
```

(Si ya las tienes, te lo indicará.)

**3.2 Instalar Homebrew** (si aún no lo tienes):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Al terminar, Homebrew muestra 2–3 comandos “Next steps” para añadirlo al PATH: **cópialos y ejecútalos**. Comprueba con `brew --version`.

**3.3 Instalar Python 3.12 y tkinter:**

```bash
brew install python@3.12 python-tk@3.12
```

**3.4 Crear la carpeta del taller y el entorno virtual:**

```bash
mkdir taller_fnn
cd taller_fnn
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Comprueba con `python --version` (debe decir 3.12.x).

## 4. Instalar las librerías

Con el entorno virtual **activado** (debes ver `(venv)` al inicio de la línea):

**Mac con Apple Silicon (M1/M2/M3/M4):**

```bash
pip install tensorflow scikit-learn matplotlib
```

No necesitas `tensorflow-metal` (aceleración por GPU) para este taller.

**Mac con Intel:** las versiones recientes de TensorFlow ya no publican paquetes para macOS Intel; la última compatible es la 2.16. Usa:

```bash
pip install "tensorflow==2.16.2" scikit-learn matplotlib
```

> ¿No sabes cuál tienes? Ejecuta `uname -m`: `arm64` = Apple Silicon, `x86_64` = Intel.

La descarga es grande (varios cientos de MB); puede tardar unos minutos.

Crea un archivo `requirements.txt` (opcional, para reproducir el entorno después):

```text
# Apple Silicon:
tensorflow
scikit-learn
matplotlib

# Mac Intel: cambia la primera línea por tensorflow==2.16.2
```

## 5. Verificar la instalación

```bash
python -c "import tensorflow as tf, sklearn, matplotlib; print(tf.__version__); print('Todo OK')"
```

Debe imprimir la versión de TensorFlow (por ejemplo `2.x.x`) y `Todo OK`. Si aparece un error, ve a la sección **Solución de problemas**.

## 6. Crear el script

Crea el archivo `train_iris.py` dentro de la carpeta `taller_fnn`:

```bash
nano train_iris.py
```

Pega el código, guarda con `Ctrl+O`, `Enter` y sal con `Ctrl+X`. (Alternativas: `open -e train_iris.py` abre TextEdit —usa *Formato → Convertir a texto plano*— o VS Code con `code .`).

Pega este código completo:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # menos mensajes de TensorFlow

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

# ---------- 1. CARGAR DATOS ----------
iris = load_iris()
X, y = iris.data, iris.target          # X: (150, 4)  y: 0, 1, 2
nombres = iris.target_names            # setosa, versicolor, virginica
print("Features:", iris.feature_names)
print("Forma de X:", X.shape)

# ---------- 2. DIVIDIR Y NORMALIZAR ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # aprende media/desviación SOLO con train
X_test = scaler.transform(X_test)         # aplica lo mismo a test

# ---------- 3. CONSTRUIR LA RED ----------
model = keras.Sequential([
    keras.Input(shape=(4,)),                # capa de entrada: 4 medidas de la flor
    layers.Dense(16, activation="relu"),    # oculta 1
    layers.Dense(8, activation="relu"),     # oculta 2
    layers.Dense(3, activation="softmax"),  # salida: probabilidad de 3 especies
])
model.summary()

# ---------- 4. COMPILAR ----------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------- 5. ENTRENAR ----------
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=16,
    validation_split=0.2, verbose=0,
)
for ep in range(0, 100, 20):
    print(f"Época {ep+1:3d} | loss={history.history['loss'][ep]:.3f} "
          f"| val_acc={history.history['val_accuracy'][ep]:.3f}")

# ---------- 6. EVALUAR ----------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en test: {test_acc:.3f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=nombres))

# ---------- 7. PREDECIR UNA FLOR NUEVA ----------
flor = np.array([[5.1, 3.5, 1.4, 0.2]])   # sépalo largo/ancho, pétalo largo/ancho (cm)
probs = model.predict(scaler.transform(flor), verbose=0)[0]
print("Probabilidades:", dict(zip(nombres, probs.round(3))))
print("Predicción:", nombres[np.argmax(probs)])

# ---------- 8. GRÁFICAS ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="entrenamiento")
ax[0].plot(history.history["val_loss"], label="validación")
ax[0].set_title("Pérdida por época")
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Loss"); ax[0].legend()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=nombres).plot(ax=ax[1], colorbar=False)
ax[1].set_title("Matriz de confusión (test)")
plt.tight_layout()
plt.savefig("iris_resultados.png", dpi=120)
print("\nGráfica guardada en iris_resultados.png")
plt.show()
```

## 7. Ejecutar

```bash
python train_iris.py
```

**Salida esperada (aproximada):**

```text
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Forma de X: (150, 4)
... (resumen del modelo: ~ 300 parámetros) ...
Época   1 | loss=1.0xx | val_acc=0.6xx
...
Precisión en test: 0.93 – 1.00
Predicción: setosa
Gráfica guardada en iris_resultados.png
```

Los números exactos pueden variar un poco según tu equipo. Se generará `iris_resultados.png` con la curva de pérdida y la matriz de confusión.

## 8. ¿Qué hace cada parte? (relación con la teoría)

| Sección del código | Concepto de la FNN |
|---|---|
| 1. Cargar datos | Las 4 medidas son las **neuronas de entrada** (una por feature) |
| 2. Normalizar | Poner las features en escala similar ayuda a que el entrenamiento converja. Ajustamos el `scaler` solo con *train* para no “espiar” el test |
| 3. Construir la red | `Dense` = capa totalmente conectada: cada neurona calcula `z = w·x + b` y aplica la **activación** (`ReLU`). `softmax` en la salida convierte 3 valores en probabilidades que suman 1 |
| 4. Compilar | Define la **función de pérdida** (error) y el **optimizador** (cómo se actualizan los pesos) |
| 5. Entrenar | Cada época repite: **forward propagation → pérdida → backpropagation → actualización de pesos** |
| 6. Evaluar | Se mide en datos **nunca vistos** (test): accuracy, precision, recall, F1 |
| 7. Predecir | Solo se usa **forward propagation** |
| 8. Gráficas | Si la curva de validación sube mientras la de entrenamiento baja, hay **sobreajuste** |

## 9. Ejercicios propuestos

1. Cambia las neuronas ocultas (`16` y `8`) a `4` y `2`. ¿Baja la precisión?
2. Cambia `epochs=100` por `10`. ¿Qué pasa con la pérdida? (subajuste)
3. Cambia `activation="relu"` por `"sigmoid"` o `"tanh"`. ¿Cambia la velocidad de aprendizaje?
4. Comenta las líneas del `StandardScaler` (usa `X_train`/`X_test` sin escalar). Compara resultados.
5. Cambia `learning_rate=0.01` por `0.1` y por `0.0001`. Observa la curva de pérdida.
6. Prueba una flor tuya: `[6.5, 3.0, 5.5, 2.0]` (debería ser *virginica*).

## 10. Solución de problemas

**Específicos de macOS:**

- *`zsh: command not found: brew`*: no ejecutaste los comandos “Next steps” de Homebrew (paso 3.2). Ciérrala y ábrela de nuevo la Terminal tras hacerlo.
- *`Could not find a version that satisfies the requirement tensorflow`*: Python demasiado nuevo o Mac Intel con TensorFlow reciente. Usa Python 3.12 y, si es Intel, `tensorflow==2.16.2`.
- *`ModuleNotFoundError: No module named '_tkinter'`*: instala `brew install python-tk@3.12` y recrea el entorno virtual. Recuerda que la imagen `iris_resultados.png` se guarda igualmente.
- *`python3.12: command not found`*: reabre la Terminal o usa la ruta completa (`/opt/homebrew/bin/python3.12` en Apple Silicon, `/usr/local/bin/python3.12` en Intel).
- *No uses el Python del sistema (`/usr/bin/python3`)*: es antiguo y limitado; usa siempre el entorno virtual.

**Comunes a todos los sistemas:**

- *`ModuleNotFoundError`*: el entorno virtual no está activado o instalaste las librerías fuera de él. Actívalo y repite el paso 4.
- *Advertencias de TensorFlow (`oneDNN`, `cuda`, `TF-TRT`)*: son informativas; puedes ignorarlas. Este ejercicio corre bien solo con CPU.
- *La ventana de la gráfica no abre*: no es grave, la imagen `iris_resultados.png` se guarda igualmente en la carpeta.
- *Plan B si TensorFlow no se deja instalar*: puedes usar `MLPClassifier` de scikit-learn, que implementa también una red feedforward:

```python
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

## 11. Para desactivar el entorno y limpiar

```bash
deactivate
```

Para borrar todo, elimina la carpeta `taller_fnn`.
