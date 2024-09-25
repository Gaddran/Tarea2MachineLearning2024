#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Tomás Fontecilla </em><br>
# 
# </div>
# 
# # Machine Learning
# *28 de Septiembre de 2024*
# 
# **Nombre Estudiante(s)**: ` Nicolas Gonzalez - Giuseppe Lavarello - Camilo Rivera`  

# ## Introducción
# 
# <div style="text-align: justify"> La segmentación de imágenes es una técnica fundamental en el procesamiento de imágenes que busca dividir una imagen en regiones homogéneas. En este trabajo, exploramos la aplicación de tres algoritmos de clustering populares: K-means, jerárquico y mezclas gaussianas, para segmentar una imagen. Utilizando la biblioteca scikit-learn, comparamos la efectividad de cada algoritmo en la identificación de regiones de color similares y en la reducción de la complejidad visual de la imagen. A través del análisis de los resultados obtenidos, se busca determinar cuál de estos métodos es más adecuado para la tarea de segmentación en el contexto de la imagen seleccionada. </div>
# 

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from PIL import Image


# ### Carga de Imagen

# Cargar imagen
image_path = './data/ciudad.jpeg'
image = Image.open(image_path)
image = image.convert('RGB')  # Asegurar que la imagen está en formato RGB
image_array = np.array(image)

# Mostrar imagen original
plt.imshow(image_array)
plt.axis('off')  # Sin ejes
plt.title("Imagen original")
plt.show()

# Reformatear la imagen para el clustering (filas y columnas en un array 2D)
pixels = image_array.reshape(-1, 3)  # Flatten the image to [n_pixels, 3]

pixels.shape


# Ver las dimensiones de la imagen original
width, height = image.size
print(f"Dimensiones de la imagen: {width} x {height}")
total_pixels = width * height
print(f"Total de píxeles: {total_pixels}")


# <div style="text-align: justify"> La imagen original se ha vectorizado, resultando en una matriz de características de 320,000 x 3, donde cada fila corresponde a un píxel expresado en el espacio de color RGB. A continuación, se empleó el algoritmo K-means para realizar una cuantización de color. Mediante el análisis del gráfico de codo, se identificó el valor de K (número de clusters) que optimiza la representación de la imagen con una paleta de colores reducida.
# </div>

# ### Determinación del número óptimo de colores

# Método del codo (Scree.plot) para determinar la cantidad óptima de colores para KMeans
def calculate_inertia(pixels, max_clusters=10):
    inertia = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
        inertia.append(kmeans.inertia_)
    return inertia

# Graficar incercia para las cantidades de clusters
inertia = calculate_inertia(pixels, max_clusters=10)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Scree Plot (método del codo) para la cantidad optima de colores')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# **Desición**
# <div style="text-align: justify">El gráfico de codo revela un punto de inflexión claro alrededor de los 7 clusters. A partir de este punto, la curva experimenta una disminución gradual en la inercia, lo que sugiere que agregar más clusters no conlleva una mejora sustancial en la calidad de la segmentación. La estabilización de la curva en los últimos cuatro clusters, alrededor de una inercia de 0.4, refuerza la idea de que 7 clusters representan un equilibrio óptimo entre la complejidad del modelo y la capacidad de capturar la variabilidad de los datos. </div>

# #### KMeans y paleta de colores

# Supongamos que el número óptimo de clusters es 7
n_clusters = 7  
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)

palette = kmeans.cluster_centers_.astype(int)
# Mostrar la paleta de colores encontrada
print("Paleta de colores obtenida:", palette)


# 'palette' contiene los colores obtenidos por KMeans
# 'palette' es el array de los centros de los clusters, cada fila es un color RGB

def display_palette(palette):
    # Crear una imagen que represente la paleta de colores
    palette = np.array(palette, dtype=np.uint8)  # Asegurarnos de que los valores estén en formato correcto
    n_colors = len(palette)

    # Crear una figura de 1 fila por 'n_colors' columnas para mostrar los colores
    fig, ax = plt.subplots(1, n_colors, figsize=(n_colors, 1))

    for i in range(n_colors):
        ax[i].imshow([[palette[i]]])  # Mostrar el color i-ésimo en un cuadrado
        ax[i].axis('off')  # Quitar ejes para que solo se vean los colores

    plt.show()
# Llamar a la función con la paleta obtenida por KMeans
display_palette(palette)


# ### Reconstrucción de la imagen con Kmeans

# Asignar los nuevos colores a la imagen (reducida)
kmeans_labels = kmeans.predict(pixels)
reduced_img_kmeans = palette[kmeans_labels].reshape(image_array.shape)

# Mostrar la imagen reconstruida usando KMeans
plt.imshow(reduced_img_kmeans)
plt.title("Imagen reconstruida con KMeans")
plt.axis('off')
plt.show()


# Al aplicar K-means, se produjo una simplificación de la paleta de colores original. Cada píxel se asigna al color del centroide más cercano, (en 'direccionalidad' de sus colores no de su posición), lo que resulta en regiones homogéneas de color.
# 
# Lo que podemos observar en una imagen reconstruida:
# 
# * Pérdida de detalle: Debido a la reducción en el número de colores, se pierde detalle fino y texturas. Las transiciones entre colores son más abruptas y menos naturales. 
# 
# * Bloques de color: La imagen presenta bloques de color más uniformes, especialmente en áreas con gradientes suaves y detalles finos.  
# 
# * Efecto de posterización: La imagen puede adquirio un aspecto de póster, con grandes áreas de color sólido y contornos bien definidos.  
# 
# * Dependencia del número de clusters: El número de clusters (K) utilizado directamente influye en el nivel de detalle de la imagen reconstruida. Un K más bajo resultará en una imagen más simplificada, mientras que un K alto puede preservar más detalles.

# #### Aplicación Clustering Jerárquico

# Reducir el tamaño de la imagen
small_image = image.resize((int(image.width / 2), int(image.height / 2)))  # Cambiar el factor según sea necesario. con 2 no me corre
small_image_array = np.array(small_image)

# Aplanar la imagen reducida
small_pixels = small_image_array.reshape(-1, 3)

# Aplicar el clustering jerárquico a la imagen más pequeña
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
hierarchical_labels = hierarchical.fit_predict(small_pixels)




# Crear la nueva paleta de colores
paleta_cluster = np.vstack([small_pixels[hierarchical_labels == label][0] for label in np.unique(hierarchical_labels)])

# Mostrar la paleta de colores encontrada
print("Paleta de colores obtenida:", paleta_cluster)
display_palette(paleta_cluster)


# Restaurar la imagen pequeña con clustering jerárquico
hierarchical_image_small = paleta_cluster[hierarchical_labels].reshape(small_image_array.shape)

# Mostrar la imagen restaurada
plt.imshow(hierarchical_image_small)
plt.axis('off')
plt.title("Imagen restaurada (versión reducida) con Clustering Jerárquico")
plt.show()


# Al aplicar clustering jerárquico, se crea una jerarquía de agrupamientos, donde cada nivel representa una partición diferente de los datos. Cada píxel se asigna a un cluster basado en su similitud con otros píxeles, (nuevamente con su similitud medida en el espacio de los colores, no de su posición).
# Lo que podemos observar en la imagen reconstruida:
# 
# * Segmentación basada en similitud: Los píxeles con características similares (color, textura) tienden a agruparse, lo que resulta en regiones homogéneas.
# 
# * Jerarquía de detalles: La estructura jerárquica permite explorar diferentes niveles de detalle en la segmentación. Niveles superiores del dendrograma pueden mostrar una segmentación más gruesa, mientras que niveles inferiores pueden revelar detalles más finos.
# 
# * Poder de computo: La exigencia de un mayor poder computacional (tiempo de procesamiento y memoria RAM) obligó a aplicar una transformación a la imagen antes de su análisis, lo cual dificulta una comparación directa de los resultados con los otros metodos.
# 

# #### Aplicación Gaussian Mixture

# Ajustar el modelo Gaussian Mixture con el número óptimo de clusters
gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(pixels)

# Obtener las etiquetas
gmm_labels = gmm.predict(pixels)




# Crear la nueva paleta de colores
palette_means = np.array(gmm.means_.astype(int))

# Mostrar la paleta de colores encontrada
print("Paleta de colores obtenida:", palette_means)
display_palette(palette_means)


# Restaurar la imagen con Gaussian Mixtures
gmm_image = palette_means[gmm_labels].reshape(image_array.shape)

# Mostrar la imagen restaurada
plt.imshow(gmm_image)
plt.axis('off')
plt.title("Imagen restaurada con Gaussian Mixtures")
plt.show()


# ## Conclusiones  
# 
# ### Algoritmo Kmeans
# <div style="text-align: justify">Mediante la aplicación del algoritmo K-Means y la reducción de la paleta de colores a 7 clusters, se consiguió una segmentación de la imagen en regiones homogéneas. Este proceso no solo simplifica la imagen, sino que también realza las formas y contornos, facilitando la identificación de los diferentes elementos visuales y manteniendo intactas las características fundamentales de la imagen original.</div>
# 
# 
# ### Algoritmo Cluster jerarquicos
# <div style="text-align: justify"> El clustering jerárquico ha demostrado ser más efectivo en capturar la complejidad visual de la imagen, revelando una segmentación más detallada y rica en matices de color. Esta técnica ha destacado especialmente en áreas con transiciones suaves y variaciones sutiles, como los anuncios luminosos y las superficies reflectantes. Aunque ha preservado los detalles generales, algunos elementos específicos han perdido algo de definición.</div>
# 
# ### Algoritmo Gaussian Mixtures
# <div style="text-align: justify"> En comparación con otros métodos de segmentación, GMM ha mostrado una mayor capacidad para capturar la complejidad de las imágenes, especialmente en áreas con transiciones de color suaves y detalles finos. La segmentación resultante es más natural y realista, sin perder la precisión en la delimitación de los objetos, ademas manteniendo la estructura visual global."</div>
# 
# ### Conclusión General
# <div style="text-align: justify">Los resultados obtenidos permiten concluir que el Modelo de Mezcla Gaussianas es la técnica más adecuada para la segmentación de esta imagen. La capacidad de GMM para modelar distribuciones de color multimodales ha permitido obtener una segmentación más precisa y detallada, especialmente en áreas con variaciones de color suaves y texturas complejas. En comparación, K-means y el clustering jerárquico han mostrado limitaciones en la representación de estos detalles. </div>
# 
