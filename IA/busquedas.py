import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


# Grafo de ejemplo
grafo = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": ["H"],
    "F": [],
    "G": ["I"],
    "H": [],
    "I": []
}


def bfs(grafo, inicio, objetivo):
    cola = deque([(inicio, [inicio])])
    visitados = set()

    orden = []

    while cola:
        nodo, camino = cola.popleft()

        if nodo in visitados:
            continue

        visitados.add(nodo)
        orden.append(nodo)

        if nodo == objetivo:
            return camino, orden

        for vecino in grafo[nodo]:
            if vecino not in visitados:
                cola.append((vecino, camino + [vecino]))

    return None, orden


camino, orden = bfs(grafo, "A", "I")

print("BFS")
print("Orden de exploración:", orden)
print("Camino encontrado:", camino)


# -----------------------------
# DIAGRAMA
# -----------------------------

G = nx.DiGraph()

for nodo, vecinos in grafo.items():
    for vecino in vecinos:
        G.add_edge(nodo, vecino)

pos = {
    "A": (0, 3),
    "B": (-1, 2),
    "C": (1, 2),
    "D": (-2, 1),
    "E": (0, 1),
    "F": (0, 1),
    "G": (2, 1),
    "H": (0, 0),
    "I": (2, 0)
}

colores = []

for nodo in G.nodes:
    if nodo in camino:
        colores.append("lightgreen")
    elif nodo in orden:
        colores.append("orange")
    else:
        colores.append("lightblue")

plt.figure(figsize=(9, 6))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=colores,
    node_size=1800,
    arrows=True,
    font_size=12
)

plt.title("Búsqueda ciega BFS\nVerde = camino | Naranja = explorado")
plt.show()


# ============================================================
# 2. BÚSQUEDA HEURÍSTICA: A*
# ============================================================

import heapq


# Grafo ponderado
grafo_astar = {
    "A": {"B": 1, "C": 4},
    "B": {"D": 2, "E": 5},
    "C": {"F": 2},
    "D": {"G": 3},
    "E": {"G": 1},
    "F": {"G": 2},
    "G": {}
}


# Heurística: estimación de distancia hasta G
heuristica = {
    "A": 6,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 1,
    "F": 2,
    "G": 0
}


def a_estrella(grafo, inicio, objetivo, h):

    cola = []

    # (f, costo, nodo, camino)
    heapq.heappush(
        cola,
        (h[inicio], 0, inicio, [inicio])
    )

    costos = {inicio: 0}
    explorados = []

    while cola:

        f, costo, nodo, camino = heapq.heappop(cola)

        if nodo in explorados:
            continue

        explorados.append(nodo)

        if nodo == objetivo:
            return camino, explorados

        for vecino, peso in grafo[nodo].items():

            nuevo_costo = costo + peso

            if vecino not in costos or nuevo_costo < costos[vecino]:

                costos[vecino] = nuevo_costo

                f_nuevo = nuevo_costo + h[vecino]

                heapq.heappush(
                    cola,
                    (
                        f_nuevo,
                        nuevo_costo,
                        vecino,
                        camino + [vecino]
                    )
                )

    return None, explorados


camino_astar, explorados_astar = a_estrella(
    grafo_astar,
    "A",
    "G",
    heuristica
)

print("\nA*")
print("Nodos explorados:", explorados_astar)
print("Camino encontrado:", camino_astar)


# -----------------------------
# DIAGRAMA DE A*
# -----------------------------

G2 = nx.DiGraph()

for nodo, vecinos in grafo_astar.items():
    for vecino, peso in vecinos.items():
        G2.add_edge(nodo, vecino, weight=peso)

pos2 = {
    "A": (0, 2),
    "B": (-1, 1),
    "C": (1, 1),
    "D": (-1, 0),
    "E": (0, 0),
    "F": (2, 0),
    "G": (1, -1)
}

colores2 = []

for nodo in G2.nodes:

    if nodo in camino_astar:
        colores2.append("lightgreen")

    elif nodo in explorados_astar:
        colores2.append("orange")

    else:
        colores2.append("lightblue")


plt.figure(figsize=(9, 6))

nx.draw(
    G2,
    pos2,
    with_labels=True,
    node_color=colores2,
    node_size=1800,
    arrows=True,
    font_size=12
)

etiquetas = nx.get_edge_attributes(G2, "weight")

nx.draw_networkx_edge_labels(
    G2,
    pos2,
    edge_labels=etiquetas
)

plt.title("Búsqueda heurística A*\nVerde = camino | Naranja = explorado")
plt.show()


# ============================================================
# 3. ALGORITMO EVOLUTIVO: ALGORITMO GENÉTICO
# ============================================================

import random


# Queremos maximizar:
#
# f(x) = x²
#
# donde x está entre 0 y 31.
# Un individuo se representa con 5 bits.


def crear_individuo():
    return [
        random.randint(0, 1)
        for _ in range(5)
    ]


def convertir(individuo):
    numero = 0

    for bit in individuo:
        numero = numero * 2 + bit

    return numero


def fitness(individuo):
    x = convertir(individuo)
    return x ** 2


def seleccion(poblacion):

    participantes = random.sample(poblacion, 3)

    return max(
        participantes,
        key=fitness
    )


def crossover(padre1, padre2):

    punto = random.randint(1, 4)

    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]

    return hijo1, hijo2


def mutacion(individuo, probabilidad=0.05):

    nuevo = individuo.copy()

    for i in range(len(nuevo)):

        if random.random() < probabilidad:
            nuevo[i] = 1 - nuevo[i]

    return nuevo


def algoritmo_genetico(
    tam_poblacion=10,
    generaciones=30
):

    poblacion = [
        crear_individuo()
        for _ in range(tam_poblacion)
    ]

    historial = []

    for generacion in range(generaciones):

        nueva_poblacion = []

        while len(nueva_poblacion) < tam_poblacion:

            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)

            hijo1, hijo2 = crossover(
                padre1,
                padre2
            )

            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)

            nueva_poblacion.append(hijo1)

            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

        mejor = max(
            poblacion,
            key=fitness
        )

        historial.append(
            fitness(mejor)
        )

        print(
            f"Generación {generacion + 1}: "
            f"x = {convertir(mejor)}, "
            f"fitness = {fitness(mejor)}"
        )

    mejor = max(
        poblacion,
        key=fitness
    )

    return mejor, historial


mejor, historial = algoritmo_genetico()

print("\nAlgoritmo genético")
print("Mejor individuo:", mejor)
print("Valor de x:", convertir(mejor))
print("Fitness:", fitness(mejor))


# -----------------------------
# DIAGRAMA DEL ALGORITMO
# -----------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(historial) + 1),
    historial,
    marker="o",
    color="purple"
)

plt.title("Evolución del algoritmo genético")
plt.xlabel("Generación")
plt.ylabel("Mejor fitness")
plt.grid(True)

plt.show()