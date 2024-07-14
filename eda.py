import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import folium
import json
from module.ml_func import * 
from streamlit_folium import folium_static  


def eda_app():

    st.subheader(body = "Exploratory Data Analysis")

    st.write("En esta sección puedes realizar un análisis más detallado de los datos a través de gráficos y visualizaciones.")

    st.sidebar.markdown("*"*10)
    
    st.sidebar.markdown("Selecciona `type`, `platform`, `genres` y `directors` para explorar los datos y las gráficas.")

    df = read_eda()

    # # fig1

    fig1 = px.histogram(data_frame = df,
             x          = "type",
             title      = 'TYPES',
             color      = "type",
             nbins      = 50)
    
    fig1.update_layout()
    st.plotly_chart(figure_or_data = fig1, use_container_width = True)

    # SIDEBAR
    df_sidebar = df.sort_values(by = "release_year").copy()

    ### Model Type
    model_type_options = ["All"] + list(df_sidebar["type"].unique())
    model_type = st.sidebar.selectbox(label   = "Select type:",
                                      options = model_type_options,
                                      index   = 0)
    

    if model_type != "All":
        df_sidebar = df_sidebar[df_sidebar["type"] == model_type]


    ### Platform
    platform_options = ["All"] + list(df_sidebar["platform"].unique())
    platform_type = st.sidebar.multiselect(label   = "Select platform:",
                                         options =  platform_options,
                                         default = ["All"])
    
    if "All" in platform_type:
        df_sidebar = df_sidebar  # Si "All" está seleccionado, mostrar todos los datos
    else:
        df_sidebar = df_sidebar[df_sidebar["platform"].isin(platform_type)]

    
    ## Genre
    categorias_unicas = set()
    for genero in df_sidebar['genres'].dropna().values:
        generos = genero.split(", ")
        categorias_unicas.update(generos)
    categorias_unicas = list(categorias_unicas)
    
    genre_options = ["All"] + categorias_unicas
    genre_type = st.sidebar.multiselect(label   = "Select genre:",
                                         options = genre_options,
                                         default   = ["All"])

    if "All" not in genre_type:
        df_sidebar = df_sidebar[df_sidebar['genres'].apply(lambda x: any(g in genre_type for g in x.split(", ")))]
    

    ## directors
    directores_unicos = set()
    for director in df_sidebar['directors'].dropna().values:
        directores = director.split(", ")
        directores_unicos.update(directores)
    directores_unicos = list(directores_unicos)

    directors_options = ["All"] + directores_unicos
    directors_type = st.sidebar.multiselect(label   = "Select directors:",
                                         options =  directors_options,
                                         default = ["All"])
    if "All" in directors_type:
        df_sidebar = df_sidebar  # Si "All" está seleccionado, mostrar todos los datos
    else:
        df_sidebar = df_sidebar[df_sidebar["directors"].isin(directors_type)]
    
    df_sidebar.reset_index(drop = True, inplace = True)
    
    with st.expander(label = "DataFrame", expanded = False):
        st.dataframe(df_sidebar)
        st.write(f"DataFrame dimensions: {df_sidebar.shape[0]}x{df_sidebar.shape[1]}")

     # fig2

    duration = df_sidebar.sort_values(by='runtime', ascending=False).head(10)
    fig2 = px.bar(duration, x='title', y='runtime',color= 'title', title="Peliculas/Series Mayor Duración", labels={'title': 'Title', 'runtime': 'Minutos'})
    fig2.update_layout(width=1200, height=600)
     
    

    # fig3

    platfms = df_sidebar.groupby(['platform', 'type']).size().reset_index(name='count')
    fig3 = px.bar(
    platfms,
    x='platform',
    y='count',
    color='platform',
    title='Peliculas y Series por Plataforma',
    facet_col='type',
    facet_col_wrap=2,
    text='count')

    fig3.update_layout(
    xaxis_title='Plataforma',
    yaxis_title='Cantidad',
    title_font_size=24,
    title_x=0.5,
    legend_title_text='Plataforma',
    legend=dict(
        x=1.05,
        y=1
    ))

    fig3.update_traces(
    texttemplate='%{text}',
    textposition='outside')

    
    
    # fig4

    fig4 = px.histogram(data_frame = df_sidebar,
             x          = "age_certification",
             title      = 'Catalago de edades',
             color      = "age_certification",
             nbins      = 50)

    
    
    # fig5

    directores = df_sidebar['directors'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='director')
    directores = directores['director'].value_counts(dropna = False).reset_index().head(20)
    fig5 = px.bar(directores, x='director', y='count',color= 'director', title="Directores", labels={'director': 'Directores', 'count': 'Count'})
    fig5.update_layout(width=1200, height=600)

    # fig6

    actores = df_sidebar['actors'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='actor')
    actores = actores['actor'].value_counts(dropna = False).reset_index().head(15)
    fig6 = px.bar(actores, x='actor', y='count',color= 'actor', title="Actores", labels={'actor': 'Actores', 'count': 'Count'})
    fig6.update_layout(width=1200, height=600)

    # fig7

    mayor_votos = df_sidebar.sort_values(by='imdb_votes', ascending=False)

    fig7 = px.bar(mayor_votos.head(10), x='title', y='imdb_votes',color= 'title', title="Peliculas/Series con mayor cantidad de votos usuarios Imdb", labels={'title': 'Title', 'imdb_votes': 'Votos'})
    fig7.update_layout(width=1000, height=600)

    # fig8

    top = df_sidebar.sort_values(by='imdb_score', ascending=False)
    fig8 = px.bar(top.head(12), x='title', y='imdb_score',color= 'title', title="Peliculas /Series con Mayor Valoracion usuarios Imdb", labels={'title': 'Title', 'imdb_score': 'puntacion'})
    fig8.update_layout(width=1000, height=700)

    #fig9

    peliculas_por_ano = df_sidebar['release_year'].value_counts().sort_index().reset_index()
    peliculas_por_ano.columns = ['release_year', 'count']
    fig9 = px.line(peliculas_por_ano, x='release_year', y='count', title='Tendencia en el Número de Películas/Series por Año')
    fig9.update_layout(xaxis_title='Año de Lanzamiento', yaxis_title='Número de Películas')

    #fig10

    df_sidebar['genres'] = df_sidebar['genres'].str.replace(r'[','').str.replace(r"'",'').str.replace(r']','')
    df_sidebar['genres'] = df_sidebar['genres'].replace('', np.nan)
    generos = df_sidebar['genres'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='genero')
    generos['genero'].value_counts(dropna = False)
    
    fig10 = px.histogram(
    generos,
    x='genero',
    title='Countplot de Géneros',
    color='genero',
    color_discrete_sequence=px.colors.qualitative.Pastel)

    fig10.update_layout(
    xaxis_title='Géneros',
    yaxis_title='Conteo',
    title_font_size=24,
    title_x=0.5,
    bargap=0.2,
    width=1200,
    height=600)

    #fig11

    paises = df_sidebar['production_countries'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='pais')
    value_counts = paises['pais'].value_counts(dropna = False)
    df_paises_total = value_counts.reset_index()
    df_paises_total.columns = ['pais', 'total']
    df_paises_total['pais_completo'] = df_paises_total['pais'].replace(abreviaciones_a_nombres)
    df_paises_total['Total_log'] = df_paises_total['total'].apply(np.log)

    world_geo = "source/world_countries.json" # Archivo GeoJSON

    world_map = folium.Map(location = [0, 0], zoom_start = 2)

    folium.Choropleth(geo_data = world_geo,
                  data     = df_paises_total,
                  columns  = ["pais_completo", "Total_log"],
                  fill_color   = "YlGn",
                  key_on   = "feature.properties.name").add_to(world_map)
    
    #fig12
    
    shows = df_sidebar.sort_values(by='seasons', ascending=False).head(10)
    fig12 = px.bar(shows, x='title', y='seasons',color= 'title', title="Seires Más Temporadas", labels={'title': 'Title', 'seasons': 'seasons'})
    fig12.update_layout(width=1200, height=600)



    # Plots

    st.plotly_chart(figure_or_data = fig3, use_container_width = True)
    st.plotly_chart(figure_or_data = fig10, use_container_width = True)
    st.plotly_chart(figure_or_data = fig4, use_container_width = True)
    st.plotly_chart(figure_or_data = fig5, use_container_width = True)
    st.plotly_chart(figure_or_data = fig6, use_container_width = True)
    st.plotly_chart(figure_or_data = fig7, use_container_width = True)
    st.plotly_chart(figure_or_data = fig8, use_container_width = True)
    st.plotly_chart(figure_or_data = fig2, use_container_width = True)
    st.plotly_chart(figure_or_data = fig12, use_container_width = True)
    st.plotly_chart(figure_or_data = fig9, use_container_width = True)
    st.write("Mapa paises de producción")
    folium_static(fig = world_map, width = 1000)
    


    
    
    

if __name__ == "__eda_app__":
    eda_app()