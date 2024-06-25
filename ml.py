import streamlit as st
import numpy as np
import pandas as pd
from module.ml_func import *




def ml_app():
    
    st.subheader(body = "RECOMENDADOR")
    st.write("En esta sección puedes comprobar como funciona el recomendador, tienes que valorar un mínimo de 5 peliculas o series con una puntación entre 1-5.")
    df = read_reco()

    # if 'valoraciones_df' not in st.session_state:
    #     st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "rating"])

    # peliculas = ["Buscar película o serie"] + list(df["title"].unique())

    # choice = st.multiselect(label="Películas", options=peliculas)

    # valoracion = st.selectbox(label="Puntuación", options=['', '1', '2', '3', '4', '5'])

    # if choice and valoracion:
    #     nuevos_ratings = [{"title": pelicula, "rating": valoracion} for pelicula in choice]
    #     st.session_state['valoraciones_df'] = pd.concat([st.session_state['valoraciones_df'], pd.DataFrame(nuevos_ratings)], ignore_index=True)

    # if choice and valoracion:
    #     st.write(f"Has seleccionado las siguientes películas: {', '.join(choice)}")
    #     st.write(f"Tu valoración para estas películas es: {valoracion}")

    # if not st.session_state['valoraciones_df'].empty:
    #     st.write("Valoraciones Guardadas:")
    #     st.dataframe(st.session_state['valoraciones_df'])
    # else:
    #     st.write("Aún no has guardado ninguna valoración.")

    if 'valoraciones_df' not in st.session_state:
        st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "rating"])

    # Lista de películas
    peliculas = list(df["title"].unique())
    
    with st.form(key='rating_form'):
        # Selector de películas
        choice_peliculas = st.selectbox("Película", options=peliculas)
        
        # Selector de valoraciones
        valoracion = st.selectbox("Puntuación", options=['', '1', '2', '3', '4', '5'])
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button(label='Guardar valoración')
        
        if submit_button and choice_peliculas and valoracion:
            new_rating = pd.DataFrame([{"title": choice_peliculas, "rating": int(valoracion)}])
            st.session_state['valoraciones_df'] = pd.concat([st.session_state['valoraciones_df'], new_rating], ignore_index=True)
            st.success(f"Guardada la valoración de {valoracion} para la película {choice_peliculas}")

    # Mostrar valoraciones guardadas
    if not st.session_state['valoraciones_df'].empty:
        st.write("Valoraciones Guardadas:")
        st.dataframe(st.session_state['valoraciones_df'])
    
    # Generar recomendaciones cuando hay al menos 5 valoraciones
    if len(st.session_state['valoraciones_df']) >= 5:
        st.write("Recomendaciones basadas en tus valoraciones:")
        recomendaciones = generate_recommendations(st.session_state['valoraciones_df'], df)
        st.write("Aquí tienes algunas recomendaciones de películas para ti:")
        for i, rec in enumerate(recomendaciones):
            st.write(f"{i+1}. {rec}")






























if __name__ == "__ml_app__":
    ml_app()


