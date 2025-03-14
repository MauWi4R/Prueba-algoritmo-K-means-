import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Creación de datos
def crear_datos(n_samples=300, n_features=2, n_clusters=3, random_state=42):
    datos, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return datos

# Centroides iniciales
def inicializar_centroides(datos, k):
    indices = np.random.choice(len(datos), size=k, replace=False)
    return datos[indices]

def asignar_clusters(datos, centroides):
    distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)


# Actualizar centroides
def actualizar_centroides(datos, etiquetas, k):
    nuevos_centroides = np.array([datos[etiquetas == i].mean(axis=0) for i in range(k)])
    return nuevos_centroides


# Error Total
def calcular_error(datos, etiquetas, centroides):
    error = 0
    for i in range(len(centroides)):
        cluster_puntos = datos[etiquetas == i]
        error += np.sum((cluster_puntos - centroides[i]) ** 2)
    return error

# Algoritmo k-means
def k_means(datos, k, max_iter=100, tol=1e-4):
    centroides = inicializar_centroides(datos, k)
    for iteracion in range(max_iter):
        etiquetas = asignar_clusters(datos, centroides)  #clusters
        nuevos_centroides = actualizar_centroides(datos, etiquetas, k)  # Actualizar centroides

        if np.linalg.norm(nuevos_centroides - centroides) < tol:
            print(f"Convergencia alcanzada en {iteracion + 1} iteraciones.")
            break  # Verificar

        centroides = nuevos_centroides

    error_total = calcular_error(datos, etiquetas, centroides)
    return etiquetas, centroides, error_total

# Main
if __name__ == "__main__":

    datos = crear_datos(n_samples=300, n_clusters=4, random_state=42)
    plt.scatter(datos[:, 0], datos[:, 1], s=10, color='gray', alpha=0.6)
    plt.title("Datos Generados")
    plt.show()

    k = 4

    # Ejecutar
    etiquetas, centroides_finales, error_total = k_means(datos, k)

    # Graficación
    plt.figure(figsize=(8, 6))
    colores = ['red', 'blue', 'green', 'purple']
    for i in range(k):
        plt.scatter(datos[etiquetas == i, 0], datos[etiquetas == i, 1], s=10, color=colores[i], alpha=0.6, label=f'Cluster {i}')
    plt.scatter(centroides_finales[:, 0], centroides_finales[:, 1], s=100, c='black', marker='X', label='Centroides Finales')
    plt.legend()
    plt.title("Resultados del Algoritmo k-means")
    plt.show()

    # Grpaficar error
    print(f"Error total (suma de errores al cuadrado): {error_total:.4f}")
