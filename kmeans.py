import numpy as np
import matplotlib.pyplot as plt

def calcular_distancia(p1, p2):
    # Calcula la distancia euclidiana entre dos puntos
    return np.sqrt(np.sum((p1 - p2) ** 2))

def ejecutar_kmeans(datos, k, centroides_iniciales, max_iter=100):
    centroides = np.array(centroides_iniciales, dtype=float)
    
    for _ in range(max_iter):
        # Crear listas vacías para los clústeres
        clusteres = [[] for _ in range(k)]
        
        # Asignar cada punto al centroide más cercano
        for punto in datos:
            distancias = [calcular_distancia(punto, c) for c in centroides]
            indice_mas_cercano = np.argmin(distancias)
            clusteres[indice_mas_cercano].append(punto)
            
        # Guardar los centroides antiguos para verificar si hay cambios
        centroides_antiguos = np.copy(centroides)
        
        # Recalcular los centroides (promedio de los puntos en el clúster)
        for i in range(k):
            if len(clusteres[i]) > 0:
                centroides[i] = np.mean(clusteres[i], axis=0)
                
        # Si los centroides no cambian, el algoritmo ha terminado
        if np.all(centroides_antiguos == centroides):
            break
            
    return centroides, clusteres

def calcular_wcss(datos, k_max):
    wcss = []
    # Probamos K desde 1 hasta el valor máximo definido por el usuario
    for k in range(1, k_max + 1):
        # Para el método del codo, inicializamos centroides aleatorios o los primeros k puntos
        centroides_temp = datos[:k] 
        centroides_finales, clusteres = ejecutar_kmeans(datos, k, centroides_temp)
        
        suma_cuadrados = 0
        for i in range(k):
            for punto in clusteres[i]:
                suma_cuadrados += calcular_distancia(punto, centroides_finales[i]) ** 2
        wcss.append(suma_cuadrados)
        
    return wcss

if __name__ == "__main__":
    # 1. Pedir número de puntos
    n_puntos = int(input("Ingrese la cantidad de puntos de datos: "))
    
    # 2. Pedir coordenadas
    datos_lista = []
    for i in range(n_puntos):
        coords = input(f"Ingrese las coordenadas del punto {i+1} (ej. 2,3): ")
        x, y = map(float, coords.split(','))
        datos_lista.append([x, y])
    datos = np.array(datos_lista)
    
    # 3. Pedir K y centroides iniciales
    k = int(input("Ingrese el número de clústeres (K): "))
    centroides_lista = []
    for i in range(k):
        coords = input(f"Ingrese las coordenadas del centroide inicial {i+1} (ej. 1,1): ")
        x, y = map(float, coords.split(','))
        centroides_lista.append([x, y])
    centroides_iniciales = np.array(centroides_lista)
    
    # 4. Pedir K máximo para el codo
    k_max = int(input("Ingrese el valor máximo de K para el Método del Codo: "))

    # --- Ejecutar K-Means Principal ---
    centroides_finales, clusteres_finales = ejecutar_kmeans(datos, k, centroides_iniciales)
    
    print("\nCentroides finales tras la convergencia:")
    print(centroides_finales)

    # --- Gráfico de Clústeres (2D) ---
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    colores = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(k):
        if len(clusteres_finales[i]) > 0:
            cluster_np = np.array(clusteres_finales[i])
            plt.scatter(cluster_np[:, 0], cluster_np[:, 1], c=colores[i%len(colores)], label=f'Cluster {i+1}')
    # Dibujar centroides
    plt.scatter(centroides_finales[:, 0], centroides_finales[:, 1], c='black', marker='X', s=100, label='Centroides')
    plt.title('Gráfico de Clústeres K-Means')
    plt.legend()

    # --- Ejecutar y Graficar Método del Codo ---
    wcss = calcular_wcss(datos, k_max)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, k_max + 1), wcss, marker='o', linestyle='--')
    plt.title('Curva del Método del Codo')
    plt.xlabel('Número de Clústeres (K)')
    plt.ylabel('WCSS')
    
    plt.tight_layout()
    plt.show()