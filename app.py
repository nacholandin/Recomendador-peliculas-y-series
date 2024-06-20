import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
import os
from module.ml_func import *
from eda import eda_app
from ml import ml_app
from about import about_app
from datos import datos_app

def main():
    st.set_page_config(**PAGE_CONFIG)

    menu = ["Main App", "Datos", "Exploratory Data Analysis", "Machine Learning Model", "About"]

    choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)
 
    if choice == "Main App":
        st.header(body = "Introducción")

        st.subheader("Proyecto final para Hack a Boss")

        st.write("""Este proyecto constituye el punto culminante de nuestra formación en el Bootcamp de Data Science e Inteligencia Artificial en Hack a Boss. En nuestro tercer y último proyecto, hemos decidido perfeccionar nuestro segundo proyecto “Un recomendador de contenido de Netflix” ampliándolo con contenido de diversas plataformas y tendencias actuales para obtener recomendaciones más precisas y únicas. No sólo perfeccionando esto sino también para perder un poco de dependencia de la plataforma  y centrarnos más del usuario y las tendencias actuales.  A través de un exhaustivo análisis exploratorio de datos de dichas plataformas, este estudio ofrece recomendaciones precisas a los usuarios teniendo en cuenta el género, el director, los autores y la descripción. Cabe destacar que las recomendaciones también diferencias entre series y películas haciéndolo más preciso y personal para el usuario.""")
        
        st.subheader("Bienvenida")

        st.write("¡Bienvenidos a nuestra plataforma de análisis y recomendación de películas y series! Aquí podrás explorar datos detallados sobre usuarios, eventos y sus valoraciones.")

        st.subheader("El Equipo")

        st.write("Este estudio ha sido desarrollado por Álvaro Marcos Martín, Igor Ayestarán García, Raquel García Tajes, Ignacio Landín García. Como equipo, hemos combinado nuestras competencias técnicas y experiencia analítica para realizar un recomendador de películas y series basado en contenido. Utilizamos datos de diversas plataformas como Netflix, HBO, Amazon, Paramount y Disney, extraídos de Kaggle. Nuestro recomendador considera el género, director, actores y la descripción para ofrecer recomendaciones precisas.")

        st.write("Nuestro trabajo se centra en proporcionar un recomendador abierto en el cual puedas consultar tu próxima película o serie con un enfoque único (creado por nosotros) que no se limite a una sólo plataforma. Entrenado manualmente y sistematizado. Cada plataforma, como ya sabemos, cuenta con un recomendador pero siempre sobre su contenido, por lo que queremos ofrecer al usuario una decisión más centrada en él, abriéndolo a varias plataformas y pudiendo jugar con los géneros. El recomendador ideal sería abrirlo a todas las plataformas pero incluso podríamos abrirlo para todas si contáramos con más datos públicos de otras plataformas de la que extraer información.")
       
        st.subheader("Metodología y Objetivos")
        
        st.write("Basándonos en nuestro segundo proyecto hemos perfeccionado el recomendador (nuestro objetivo principal) ampliándolo con más plataformas y mejorando el entrenamiento y nuestro EDA. En la primera fase hemos implementado más plataformas en el recomendador. Posteriormente trabajamos para determinar la afinidad de cada usuario hacia diferentes géneros de películas en base a sus valoraciones previas. Después hemos personalizado las recomendaciones calculando los pesos, normalizándolos y aplicando una penalización a los géneros con bajas valoraciones para evitar sesgos y por último hemos Implementado el Algoritmo de Recomendación integrando lo anterior en el algoritmo de recomendación para mejorar la precisión y relevancia de las recomendaciones proporcionadas a los usuarios.")

        st.subheader("Herramientas y Tecnologías")

        st.write("Para realizar este proyecto hemos utilizado una variedad de librerías entre las que encontramos Pandas, Numpy, Matpllotlib, Seaborn, Ploty, Klinter, NLTK y Folium. Estas librerías nos han servido de base para analizar los datos y el lenguaje así como para generar visualizaciones interactivas de los resultados, tendencias y patrones. Mediante el uso de Streamlit, presentamos nuestros resultados de forma dinámica para que así los usuarios puedan interactuar con ella y de esta forma también ir perfeccionando el recomendador.")

    elif choice == "Datos":
        datos_app()

    elif choice == "Exploratory Data Analysis":
        eda_app()

    elif choice == "Machine Learning Model":
        ml_app()

    else:
        about_app()

if __name__ == "__main__":
    main()