import streamlit as st
import numpy as np
import pandas as pd
from module.ml_func import *




def ml_app():
    
    st.subheader(body = "RECOMENDADOR")
    st.write("En esta sección puedes comprobar como funciona el recomendador, tienes que valorar un mínimo de 5 peliculas o series con una puntación entre 1-5.")
    df = read_data()
    # if 'valoraciones_df' not in st.session_state:
    #     st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "rating"])
    # peliculas =["Buscar pelicula o serie"] + list(df["title"].unique())
    # choice_peliculas = st.multiselect(label = "Peliculas", options = peliculas)
    
    # valoracion = [""] + ['1','2','3','4','5']
    # choice = st.selectbox(label = "Puntación", options = valoracion)
    
    # if choice_peliculas and valoracion:
    #     for pelicula in choice:
    #         st.session_state['valoraciones_df'] = st.session_state['valoraciones_df'].append({"title": pelicula, "rating": valoracion}, ignore_index=True)
    
    # if choice_peliculas and valoracion:
    #     st.write(f"Has seleccionado las siguientes películas: {', '.join(choice)}")
    #     st.write(f"Tu valoración para estas películas es: {valoracion}")

    # if not st.session_state['valoraciones_df'].empty:
    #     st.write("Valoraciones Guardadas:")
    #     st.dataframe(st.session_state['valoraciones_df'])
    # else:
    #      st.write("Aún no has guardado ninguna valoración.")

    if 'valoraciones_df' not in st.session_state:
        st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "rating"])

    peliculas = ["Buscar película o serie"] + list(df["title"].unique())

    choice = st.multiselect(label="Películas", options=peliculas)

    valoracion = st.selectbox(label="Puntuación", options=['', '1', '2', '3', '4', '5'])

    if choice and valoracion:
        nuevos_ratings = [{"title": pelicula, "rating": valoracion} for pelicula in choice]
        st.session_state['valoraciones_df'] = pd.concat([st.session_state['valoraciones_df'], pd.DataFrame(nuevos_ratings)], ignore_index=True)

    if choice and valoracion:
        st.write(f"Has seleccionado las siguientes películas: {', '.join(choice)}")
        st.write(f"Tu valoración para estas películas es: {valoracion}")

    if not st.session_state['valoraciones_df'].empty:
        st.write("Valoraciones Guardadas:")
        st.dataframe(st.session_state['valoraciones_df'])
    else:
        st.write("Aún no has guardado ninguna valoración.")































if __name__ == "__ml_app__":
    ml_app()


