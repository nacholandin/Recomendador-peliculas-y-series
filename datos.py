import streamlit as st


def datos_app():

    st.header("Extracción de Datos y Explicación")

    st.subheader("Extracción de Datos")

    st.write("Los datos utilizados en este proyecto fueron obtenidos mediante la descarga de archivos CSV proporcionados por una fuente confiable Kaggle. La recopilación de datos fue meticulosa asegurándonos que la información venía de las fuentes originales y así conseguir un análisis preciso.")
    st.write("Uno de los mayores desafíos al iniciar un proyecto de análisis de datos es la búsqueda y adquisición de datos relevantes y confiables. Realizamos una investigación exhaustiva en diversas fuentes de datos, incluyendo repositorios populares como Kaggle y otras plataformas de datos abiertos. Finalmente, nos decidimos por el análisis de las preferencias cinematográficas, debido a la disponibilidad de datos y la relevancia del tema en el contexto actual del entretenimiento.")

    st.markdown("Metodología")

    st.write ("Nuestro enfoque se centró en proporcionar una plataforma de recomendaciones personalizadas utilizando un algoritmo basado en las valoraciones de los usuarios. A continuación, se describe el proceso de extracción y preparación de datos:")
    st.write("-Obtención de Datos:")
    st.write("Los datos fueron obtenidos de archivos CSV que contienen información detallada sobre películas, valoraciones de usuarios y características asociadas (géneros, directores, actores, etc.).")

if __name__ == "__datos_app__":
    datos_app()