import streamlit as st


def informacion_app():

    st.header("Información sobre el proyecto")

    st.write("Aquí encontrarás respuestas a algunas preguntas frecuentes sobre nuestro proyecto.")

    st.write("¿Cuál es el objetivo de este proyecto?")

    st.write("El objetivo principal de este proyecto es desarrollar un sistema de recomendaciones personalizadas de películas basado en las valoraciones de los usuarios. Utilizando técnicas de análisis de datos y modelado, buscamos proporcionar recomendaciones precisas y relevantes que mejoren la experiencia de visualización de los usuarios.")

    st.write("¿Qué tecnologías se utilizan en el proyecto?")
    
    st.write("Para el desarrollo de este proyecto, hemos utilizado una variedad de tecnologías y librerías, entre las cuales destacan:")
    st.write("- Pandas: Para la manipulación y análisis de datos.")
    st.write("- NumPy: Para realizar operaciones numéricas avanzadas.")
    st.write("- Matplotlib: Para la creación de gráficos y visualizaciones básicas.")
    st.write('- Seaborn: Para visualizaciones estadísticas más avanzadas.')
    st.write("- NLTK: Para el procesamiento del lenguaje natural.")
    st.write("- Streamlit: Para el desarrollo de la interfaz web interactiva.")
    st.write("- Plotly: Para visualizaciones interactivas y dinámicas.")
    st.write("- Folium: Para la visualización de datos geoespaciales en mapas interactivos.")

    st.write("¿Qué desafíos nos hemos encontrado?")
    st.write("Uno de los principales desafíos ha sido la integración y limpieza de datos provenientes de múltiples fuentes. Además, ajustar el algoritmo de recomendación para que sea lo suficientemente preciso y relevante también ha representado un reto significativo.")

    st.write("¿Cuáles han sido los problemas más importantes que nos hemos encontrado?")
    st.write("- Calidad de Datos: Manejar valores nulos y datos inconsistentes ha sido un reto.")    
    st.write("- Normalización de Datos: Asegurar que los pesos calculados para las recomendaciones estén adecuadamente normalizados.")
    st.write("- Penalización de Géneros: Aplicar penalizaciones adecuadas a géneros con bajas valoraciones sin introducir sesgos. ")

    st.write("¿Cómo ha sido el desarrollo con Streamlit?")
    st.write("El desarrollo con Streamlit ha sido bastante positivo. Streamlit nos ha permitido crear una interfaz de usuario interactiva de manera rápida y eficiente. Su facilidad de uso y la capacidad de actualizar y visualizar datos en tiempo real han sido cruciales para la iteración rápida y la presentación de resultados de nuestro proyecto.")

    st.write("¿Qué tipo de fortalezas destacamos en la ejecución del proyecto?")
    st.write("- Interdisciplinariedad: La combinación de habilidades en análisis de datos, programación y diseño de interfaces de usuario.")
    st.write("- Innovación: Implementación de técnicas avanzadas de procesamiento de datos y modelado.")
    st.write("- Colaboración: Trabajo en equipo eficaz y comunicación clara.")

    st.write("Autocrítica con el proyecto")
    st.write("Aunque hemos alcanzado muchos de nuestros objetivos, siempre hay áreas para mejorar. Podríamos haber optimizado mejor el preprocesamiento de datos y explorado más técnicas avanzadas de modelado. También deberíamos haber dedicado más tiempo a pruebas exhaustivas y validación de nuestros resultados.")

    st.write("¿Qué lecciones hemos aprendido con el proyecto?")
    st.write("- Importancia de la Calidad de Datos: Datos limpios y bien estructurados son fundamentales para obtener resultados precisos.")
    st.write("- Iteración y Pruebas: La importancia de iterar rápidamente y probar exhaustivamente cada componente del sistema.")
    st.write("- Colaboración Efectiva: Trabajar en equipo y comunicar de manera clara y constante es esencial para el éxito del proyecto.")

    st.write("Conclusión final del proyecto")
    st.write("Este proyecto ha sido una experiencia enriquecedora que nos ha permitido aplicar y mejorar nuestras habilidades en ciencia de datos, programación y desarrollo de interfaces. Hemos logrado crear un sistema de recomendaciones personalizado que puede ser de gran utilidad para mejorar la experiencia de los usuarios en la selección de películas.")

    st.write("¿Cómo puedo contribuir al proyecto?")
    st.write("Si estás interesado en contribuir al proyecto, puedes:")
    st.write("- Revisar y Mejorar el Código: Colabora en nuestro repositorio de GitHub revisando el código, sugiriendo mejoras o corrigiendo errores.")
    st.write("- Aportar Datos Adicionales: Ayúdanos a expandir nuestra base de datos con nuevas fuentes de datos relevantes.")
    st.write("- Probar y Validar: Realiza pruebas exhaustivas y proporciona retroalimentación para mejorar el sistema de recomendaciones.")
    st.write("- Desarrollar Nuevas Funcionalidades: Propon y desarrolla nuevas características que puedan enriquecer la experiencia del usuario.")
    st.write("Tu participación y colaboración serán altamente valoradas y contribuirán al continuo mejoramiento y éxito de nuestro proyecto.")


if __name__ == "__informacion_app__":
    informacion_app()