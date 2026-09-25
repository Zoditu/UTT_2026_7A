"""
ML + Deep Learning Lab
======================

Ejemplos incluidos:

1. Regresión lineal
2. Clasificación con árbol de decisión
3. Red neuronal sencilla
4. Reconocimiento de números MNIST

Instalación:

    pip install numpy matplotlib scikit-learn tensorflow

Ejecutar:

    python ml_lab.py
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

import tensorflow as tf


# ============================================================
# UTILIDADES
# ============================================================

def pausa():
    """Espera antes de continuar con la siguiente visualización."""
    input("\nPresiona ENTER para continuar...")


# ============================================================
# 1. REGRESIÓN LINEAL
# ============================================================

def regresion_lineal():
    print("\n" + "=" * 60)
    print("1. REGRESIÓN LINEAL")
    print("=" * 60)

    # Datos de entrenamiento
    X = np.array([[50], [70], [90], [110], [130]])
    y = np.array([800, 1100, 1400, 1700, 2000])

    # Crear y entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Predicciones
    predicciones = modelo.predict(X)

    # Nueva predicción
    nueva_casa = np.array([[100]])
    precio = modelo.predict(nueva_casa)[0]

    print(f"Pendiente aprendida: {modelo.coef_[0]:.2f}")
    print(f"Intersección: {modelo.intercept_:.2f}")
    print(f"\nUna casa de 100 m² cuesta aproximadamente:")
    print(f"${precio:.2f} mil pesos")

    # Crear puntos para dibujar la recta
    x_linea = np.linspace(40, 140, 100).reshape(-1, 1)
    y_linea = modelo.predict(x_linea)

    # Visualización
    plt.figure(figsize=(10, 6))

    plt.scatter(
        X,
        y,
        color="blue",
        s=100,
        label="Datos reales"
    )

    plt.plot(
        x_linea,
        y_linea,
        color="red",
        linewidth=3,
        label="Recta aprendida"
    )

    plt.scatter(
        nueva_casa,
        precio,
        color="green",
        s=200,
        marker="*",
        label=f"Predicción: {precio:.0f}"
    )

    plt.title("¿Cómo funciona la regresión lineal?")
    plt.xlabel("Tamaño de la casa (m²)")
    plt.ylabel("Precio (miles de pesos)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.text(
        45,
        1900,
        f"y = {modelo.coef_[0]:.2f}x + {modelo.intercept_:.2f}",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="black"
        )
    )

    plt.show()

    print("""
EXPLICACIÓN:

El modelo intenta encontrar una recta:

    y = mx + b

donde:

    x = tamaño de la casa
    y = precio
    m = pendiente
    b = intersección

El algoritmo modifica m y b para que las predicciones
estén lo más cerca posible de los datos reales.

En otras palabras:

       DATOS
         ↓
    ┌─────────┐
    │   ML    │
    └─────────┘
         ↓
   encuentra una
      función
         ↓
    y = mx + b
         ↓
    PREDICCIÓN
""")

    pausa()


# ============================================================
# 2. CLASIFICACIÓN CON ÁRBOL DE DECISIÓN
# ============================================================

def clasificacion_arbol():
    print("\n" + "=" * 60)
    print("2. CLASIFICACIÓN CON ÁRBOL DE DECISIÓN")
    print("=" * 60)

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    modelo = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    precision = accuracy_score(
        y_test,
        predicciones
    )

    print(f"Precisión: {precision:.2%}")

    # Visualización del árbol
    plt.figure(figsize=(18, 9))

    plot_tree(
        modelo,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True
    )

    plt.title("Árbol de decisión")
    plt.show()

    print("""
EXPLICACIÓN:

El árbol realiza preguntas sobre los datos.

Por ejemplo:

¿El pétalo mide menos de X?
        │
     ┌──┴──┐
    SÍ    NO
    │      │
    ↓      ↓
 clase   otra pregunta

El modelo aprende automáticamente qué preguntas
permiten separar mejor las diferentes clases.

Este ejemplo clasifica flores Iris.
""")

    pausa()


# ============================================================
# 3. RED NEURONAL SENCILLA
# ============================================================

def red_neuronal():
    print("\n" + "=" * 60)
    print("3. RED NEURONAL")
    print("=" * 60)

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Red neuronal
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nArquitectura:\n")

    modelo.summary()

    # Entrenamiento
    historial = modelo.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        verbose=0
    )

    perdida, precision = modelo.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nPrecisión final: {precision:.2%}")

    # Visualizar entrenamiento
    plt.figure(figsize=(10, 5))

    plt.plot(
        historial.history["accuracy"],
        label="Entrenamiento"
    )

    plt.plot(
        historial.history["val_accuracy"],
        label="Validación"
    )

    plt.title("Aprendizaje de la red neuronal")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

    print("""
EXPLICACIÓN:

Una neurona recibe entradas:

       x1 ───┐
       x2 ───┤
       x3 ───┼──→ NEURONA ──→ salida
       x4 ───┘

Cada entrada tiene un peso:

    salida = activación(
        x1*w1 +
        x2*w2 +
        x3*w3 +
        x4*w4 +
        bias
    )

Durante el entrenamiento, la red modifica los pesos.

Al principio:

    pesos → prácticamente aleatorios
    predicciones → malas

Después de entrenar:

    pesos → ajustados
    predicciones → mejores

Eso es aprendizaje.
""")

    pausa()


# ============================================================
# 4. DEEP LEARNING CON MNIST
# ============================================================

def deep_learning_mnist():
    print("\n" + "=" * 60)
    print("4. DEEP LEARNING - MNIST")
    print("=" * 60)

    print("\nDescargando/cargando MNIST...")

    (X_train, y_train), (X_test, y_test) = \
        tf.keras.datasets.mnist.load_data()

    # Normalizar
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Red neuronal
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entrenar
    historial = modelo.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluar
    perdida, precision = modelo.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nPrecisión en datos de prueba: {precision:.2%}")

    # Elegir imágenes
    indices = np.random.choice(
        len(X_test),
        12,
        replace=False
    )

    predicciones = modelo.predict(
        X_test[indices],
        verbose=0
    )

    clases_predichas = np.argmax(
        predicciones,
        axis=1
    )

    # Visualización
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10, 8)
    )

    for ax, indice, prediccion in zip(
        axes.ravel(),
        indices,
        clases_predichas
    ):
        ax.imshow(
            X_test[indice],
            cmap="gray"
        )

        real = y_test[indice]

        color = "green" if real == prediccion else "red"

        ax.set_title(
            f"Real: {real} | ML: {prediccion}",
            color=color
        )

        ax.axis("off")

    plt.suptitle(
        "Deep Learning: reconocimiento de números",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    # Gráfica del aprendizaje
    plt.figure(figsize=(10, 5))

    plt.plot(
        historial.history["accuracy"],
        marker="o",
        label="Entrenamiento"
    )

    plt.plot(
        historial.history["val_accuracy"],
        marker="o",
        label="Validación"
    )

    plt.title("Evolución del aprendizaje")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

    print("""
EXPLICACIÓN:

MNIST contiene imágenes de números escritos a mano.

Cada imagen:

    28 x 28 píxeles

La red hace:

    Imagen
      ↓
    Flatten
      ↓
    784 valores
      ↓
    Dense(128)
      ↓
    Dense(64)
      ↓
    Dense(10)
      ↓
    Probabilidades
      ↓
    0 1 2 3 4 5 6 7 8 9

Por ejemplo:

    [0.01, 0.02, 0.01, 0.03, 0.01,
     0.05, 0.02, 0.80, 0.03, 0.02]

La probabilidad más alta es 7.

Por eso la red dice:

    "Creo que esta imagen es un 7".
""")


# ============================================================
# MENÚ PRINCIPAL
# ============================================================

def main():

    print("""
╔════════════════════════════════════════════════════╗
║             PYTHON ML + DEEP LEARNING              ║
║                  LABORATORIO                       ║
╚════════════════════════════════════════════════════╝

Este programa contiene cuatro ejemplos:

1. Regresión lineal
2. Clasificación con árbol de decisión
3. Red neuronal
4. Deep Learning con MNIST
""")

    while True:

        print("""
¿Qué quieres ejecutar?

1 → Regresión lineal
2 → Árbol de decisión
3 → Red neuronal
4 → Deep Learning MNIST
5 → Ejecutar TODO
0 → Salir
""")

        opcion = input("Opción: ")

        if opcion == "1":
            regresion_lineal()

        elif opcion == "2":
            clasificacion_arbol()

        elif opcion == "3":
            red_neuronal()

        elif opcion == "4":
            deep_learning_mnist()

        elif opcion == "5":
            regresion_lineal()
            clasificacion_arbol()
            red_neuronal()
            deep_learning_mnist()

        elif opcion == "0":
            print("\n¡Hasta luego!")
            break

        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
