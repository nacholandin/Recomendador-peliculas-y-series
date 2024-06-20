import streamlit as st


def datos_app():

    st.header("Extracción de Datos y Explicación")

    st.subheader("Extracción de Datos")

    st.write("Los datos utilizados en este proyecto fueron obtenidos mediante la descarga de archivos CSV proporcionados por una fuente confiable Kaggle. La recopilación de datos fue meticulosa asegurándonos que la información venía de las fuentes originales y así conseguir un análisis preciso.")
    st.write("Uno de los mayores desafíos al iniciar un proyecto de análisis de datos es la búsqueda y adquisición de datos relevantes y confiables. Realizamos una investigación exhaustiva en diversas fuentes de datos, incluyendo repositorios populares como Kaggle y otras plataformas de datos abiertos. Finalmente, nos decidimos por el análisis de las preferencias cinematográficas, debido a la disponibilidad de datos y la relevancia del tema en el contexto actual del entretenimiento.")

    st.subheader("Metodología")

    st.write ("Nuestro enfoque se centró en proporcionar una plataforma de recomendaciones personalizadas utilizando un algoritmo basado en las valoraciones de los usuarios. A continuación, se describe el proceso de extracción y preparación de datos:")
    st.write("1. Obtención de Datos:")
    st.write("Los datos fueron obtenidos de archivos CSV que contienen información detallada sobre películas, valoraciones de usuarios y características asociadas (géneros, directores, actores, etc.).")
    st.write("2. Preprocesamiento de Datos:")
    st.write("- Carga de Datos: Utilizamos la librería Pandas para cargar los datos desde archivos CSV.")
    st.write("- Limpieza de Datos: Realizamos tareas de limpieza para manejar valores nulos y eliminar duplicados.")
    st.write("- Transformación de Datos: Estandarizamos las columnas y preparamos los datos para el análisis, incluyendo la transformación de textos en categorías.")
    st.write("- Generación de Pesos por Género:")
    st.write("* Cálculo de Pesos: Para cada usuario, calculamos la afinidad hacia diferentes géneros de películas basándonos en sus valoraciones previas.")
    st.write("* Normalización: Los pesos fueron normalizados para que la suma total de los pesos de un usuario fuera igual a 1.")
    st.write("* Penalización: Se aplicó una penalización a los géneros con bajas valoraciones para ajustar los pesos adecuadamente.")
    st.write("3. Implementación del Algoritmo de Recomendación:")
    st.write("Utilizamos los pesos normalizados y penalizados para personalizar las recomendaciones de películas para cada usuario.")

    st.subheader("Tecnologías Utilizadas")

    st.write("- Pandas: Manipulación y análisis de datos, carga de datos desde archivos CSV, manejo de DataFrames, y operaciones de agrupamiento y agregación.")
    st.write("- NumPy: Operaciones numéricas y manejo de arrays, realización de operaciones matemáticas, aplicación de condiciones y penalizaciones en los datos.")
    st.write("- Matplotlib: Creación de gráficos y visualizaciones básicas como histogramas y gráficos de barras.")
    st.write("- Seaborn: Visualización de datos estadísticos, creación de gráficos de barras, gráficos de distribución y otras visualizaciones estadísticas.")
    st.write("- Plotly: Creación de visualizaciones de datos interactivas y dinámicas, generación de gráficos avanzados.")
    st.write("- Folium: Visualización de datos geoespaciales, creación de mapas interactivos.")
    st.write("- NLTK: Procesamiento del lenguaje natural, tokenización, lematización y eliminación de palabras vacías para preprocesamiento de texto.")
    st.write("- Streamlit: Desarrollo de aplicaciones web interactivas para la presentación de proyectos, estructuración y presentación de contenido textual, gráficos y recolección de entradas del usuario.")

    st.subheader("Fuente de los Datos")
    st.write("Los datos para este estudio fueron obtenidos de una base de datos de películas y valoraciones de usuarios. Esta base de datos es ampliamente utilizada en la industria del entretenimiento para analizar las preferencias de los espectadores y proporcionar recomendaciones personalizadas.")

    st.subheader("Uso de los Datos")
    st.write("Los datos recopilados permiten realizar un análisis detallado de las preferencias cinematográficas de los usuarios, identificar patrones en las valoraciones, y generar recomendaciones personalizadas para mejorar la experiencia de visualización.")

    st.subheader("Compromiso Futuro")
    st.write("Mirando hacia el futuro, nuestro compromiso es expandir y mejorar este proyecto incorporando más datos y ajustando nuestros modelos para aumentar la precisión de las recomendaciones. Planeamos integrar datos adicionales y explorar nuevas técnicas de modelado para proporcionar insights más profundos y relevantes.")
    st.write("Esta metodología y el uso de tecnologías avanzadas nos permiten ofrecer soluciones innovadoras en el campo de las recomendaciones personalizadas, contribuyendo significativamente a mejorar la experiencia del usuario en la industria del entretenimiento.")





if __name__ == "__datos_app__":
    datos_app()