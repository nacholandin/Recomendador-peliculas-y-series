import streamlit as st
import pandas as pd
import os
import pickle
import base64
#import sklearn

PAGE_CONFIG = {"page_title"             : "Recomendator Shows and Movies Model - Streamlit"}
                # "page_icon"             : ":robot_face:",
                # "layout"                : "wide",
                # "initial_sidebar_state" : "expanded"}

categorias_unicas = list(['horror',
 'action',
 'music',
 'crime',
 'animation',
 'fantasy',
 'western',
 'scifi',
 'comedy',
 'reality',
 'sport',
 'drama',
 'war',
 'thriller',
 'history',
 'european',
 'documentation',
 'desconocido',
 'romance',
 'family'])

@st.cache_data
def read_data():

    df = pd.read_csv("source/df.csv")

    df.columns = ['id', 'title', 'type', 'description', 'release_year',
       'age_certification', 'runtime', 'genres', 'production_countries',
       'seasons', 'imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity',
       'tmdb_score', 'actors', 'actor_ids', 'directors', 'director_ids',
       'platform', 'titleyear']
    
    df = df.drop(['id', 'description','imdb_id','tmdb_popularity','tmdb_score','actor_ids','director_ids', 'titleyear'], axis=1)

    df['genres'] = df['genres'].str.replace(r'[','').str.replace(r"'",'').str.replace(r']','')

    df['genres'] = df['genres'].replace('', 'desconocido')


    return df

