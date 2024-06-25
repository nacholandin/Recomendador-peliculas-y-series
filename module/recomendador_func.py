# import streamlit as st
# import numpy as np
# import pandas as pd
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # new_rating = pd.DataFrame([{"title": choice_peliculas, "rating": int(valoracion)}])
# # st.session_state['valoraciones_df'] = pd.concat([st.session_state['valoraciones_df'], new_rating], ignore_index=True)
# if 'valoraciones_df' not in st.session_state:
#         st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "Rating"])

# if 'valoraciones_df' in st.session_state:
#     df = st.session_state['valoraciones_df']


# def read_reco():
    
#     return pd.read_csv('source/cartelera.csv')
# dfre = read_reco()
# dfme = pd.merge(left = dfre, right= df, left_on= 'title', right_on= 'title', how= 'inner')

# def obtener_categorias_unicas(dfre, columna_generos='genres'):
#     categorias_unicas = set()
#     for genero in dfre[columna_generos].dropna().values:
#         generos = genero.strip("[]").replace("'", "").replace('"', "").split(", ")
#         generos = [g.strip().lower() for g in generos]
#         categorias_unicas.update(generos)
#     return list(categorias_unicas)

# def crear_matriz_generos(dfme, categorias_unicas, columna_generos='genres', columna_rating='Rating'):
#     datos = []
#     for row in dfme[dfme[columna_rating] > 0][columna_generos].dropna().values:
#         categorias_peliculas = []
#         row_generos = row.strip("[]").replace("'", "").replace('"', "").split(", ")
#         row_generos = [g.strip().lower() for g in row_generos]
#         for cat in categorias_unicas:
#             categorias_peliculas.append(int(cat in row_generos))
#         datos.append(categorias_peliculas)
#     return pd.DataFrame(data=datos, columns=list(categorias_unicas))

# def crear_matriz_ponderada(dfme, df_generos_peliculas, categorias_unicas, columna_rating='Rating'):
#     weighted_genre_matrix = pd.concat([dfme, df_generos_peliculas], axis=1)
#     weighted_genre_matrix = (weighted_genre_matrix[categorias_unicas].values.T * weighted_genre_matrix[columna_rating].values).T
#     return pd.DataFrame(weighted_genre_matrix, columns=categorias_unicas)

# def calcular_pesos_usuario(weighted_genre_matrix):
#     usuario_pesos = weighted_genre_matrix.sum()
#     return usuario_pesos / usuario_pesos.sum()

# def aplicar_penalizacion(usuario_pesos, factor_penalizacion=0.7):
#     umbral_penalizacion = usuario_pesos.sort_values(ascending=False).iloc[3]
#     usuario_pesos_penalizados = usuario_pesos.apply(lambda x: (x - umbral_penalizacion) * factor_penalizacion if x < umbral_penalizacion else x)
#     return usuario_pesos_penalizados

# categorias_unicas = obtener_categorias_unicas(dfre)
# df_generos_peliculas = crear_matriz_generos(dfme, categorias_unicas)
# dfme.reset_index(drop=True, inplace=True)
# weighted_genre_matrix = crear_matriz_ponderada(dfme, df_generos_peliculas, categorias_unicas)
# usuario_pesos = calcular_pesos_usuario(weighted_genre_matrix)
# usuario_pesos_penalizados = aplicar_penalizacion(usuario_pesos)

# def calcular_puntuacion_generos(dfre, df_generos_peliculas, usuario_pesos_penalizados, categorias_unicas):
#     dfre2 = pd.concat([dfre, df_generos_peliculas], axis=1)
#     dfre['Puntuacionge'] = (dfre2[categorias_unicas] * usuario_pesos_penalizados).sum(axis=1)
#     return dfre

# df = calcular_puntuacion_generos(dfre, df_generos_peliculas, usuario_pesos_penalizados, categorias_unicas)

# def obtener_directores_unicos(df, columna_directores='director_ids'):
#     df = df.dropna(subset=[columna_directores])
#     df.reset_index(drop=True, inplace=True)
#     return df[columna_directores].unique().tolist()

# def crear_matriz_directores(df, directores, columna_directores='director_ids'):
#     datos_directores = []
#     for row in df[columna_directores].values:
#         c_directores = [1 if director == row else 0 for director in directores]
#         datos_directores.append(c_directores)
#     return pd.DataFrame(data=datos_directores, columns=directores)

# def crear_matriz_ponderada_directores(df, df_directores, directores, columna_rating='Rating'):
#     weighted_genre_matrix2 = pd.concat([df, df_directores], axis=1)
#     weighted_genre_matrix2 = (weighted_genre_matrix2[directores].values.T * weighted_genre_matrix2[columna_rating].values).T
#     return pd.DataFrame(weighted_genre_matrix2, columns=directores)

# def calcular_pesos_directores(weighted_genre_matrix2, factor_penalizacion2=1.35):
#     usuario_d = weighted_genre_matrix2.sum()
#     usuario_d = usuario_d / usuario_d.sum()
#     umbral_penalizacion2 = usuario_d.sort_values(ascending=False).iloc[1]
#     usuario_dp = usuario_d.apply(lambda x: x * factor_penalizacion2 if x >= umbral_penalizacion2 else x)
#     return usuario_dp

# def calcular_puntuacion_directores(dfre, df_directores, usuario_dp, directores):
#     dfre3 = pd.concat([dfre, df_directores], axis=1)
#     dfre["Puntuaciond"] = (dfre3[directores] * usuario_dp).sum(axis=1)
#     return dfre

# directores = obtener_directores_unicos(dfme)
# df_directores_me = crear_matriz_directores(dfme, directores)
# weighted_genre_matrix2 = crear_matriz_ponderada_directores(dfme, df_directores_me, directores)
# usuario_dp = calcular_pesos_directores(weighted_genre_matrix2)
# df_directores_re = crear_matriz_directores(dfre, directores) 
# dfre = calcular_puntuacion_directores(dfre, df_directores_re, usuario_dp, directores)

# def obtener_actores_unicos(df, columna_actores='actor_ids'):
#     df[columna_actores] = df[columna_actores].fillna('').astype(str)
#     actores_unicos = set()
#     for actor in df[columna_actores].values:
#         actores = actor.split(", ")
#         actores_unicos.update(actores)
#     return list(actores_unicos)

# def crear_matriz_actores(df, actores_unicos, columna_actores='actor_ids', columna_rating=None):
#     df[columna_actores] = df[columna_actores].fillna('').astype(str)
#     datos = []
#     if columna_rating:
#         df_filtrado = df[df[columna_rating] > 0]
#     else:
#         df_filtrado = df
    
#     for row in df_filtrado[columna_actores].values:
#         actores_peliculas = [1 if actor in row.split(", ") else 0 for actor in actores_unicos]
#         datos.append(actores_peliculas)
#     return pd.DataFrame(data=datos, columns=actores_unicos)

# def crear_matriz_ponderada_actores(df, df_actores_peliculas, actores_unicos, columna_rating='Rating'):
#     weighted_genre_matrix = pd.concat([df, df_actores_peliculas], axis=1)
#     weighted_genre_matrix = (weighted_genre_matrix[actores_unicos].values.T * weighted_genre_matrix[columna_rating].values).T
#     return pd.DataFrame(weighted_genre_matrix, columns=actores_unicos)

# def calcular_pesos_actores(weighted_genre_matrix):
#     usuario_ac = weighted_genre_matrix.sum()
#     return usuario_ac / usuario_ac.sum()

# def calcular_puntuacion_actores(dfre, df_actores_peliculas, usuario_ac, actores_unicos):
#     dfre4 = pd.concat([dfre, df_actores_peliculas], axis=1)
#     dfre["Puntuacionac"] = (dfre4[actores_unicos] * usuario_ac).sum(axis=1)
#     return dfre

# actores_unicos = obtener_actores_unicos(dfme)
# df_actores_peliculas_me = crear_matriz_actores(dfme, actores_unicos, columna_rating='Rating')
# weighted_genre_matrix = crear_matriz_ponderada_actores(dfme, df_actores_peliculas_me, actores_unicos)
# usuario_ac = calcular_pesos_actores(weighted_genre_matrix)

# df_actores_peliculas_re = crear_matriz_actores(dfre, actores_unicos)
# dfre = calcular_puntuacion_actores(dfre, df_actores_peliculas_re, usuario_ac, actores_unicos)

# # Inicialización de lematizador y stop words
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# def preprocess_text(text):
#     if pd.isnull(text):
#         return ""
#     tokens = word_tokenize(str(text).lower())
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
#     tokens = [word for word in tokens if word not in stop_words]
#     return " ".join(tokens)

# def preprocesar_descripciones(df, columna='description'):
#     df['processed_description'] = df[columna].apply(preprocess_text)
#     return df

# def vectorizar_descripciones(df, columna='processed_description'):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(df[columna])
#     return tfidf_matrix, vectorizer

# def calcular_matriz_similitud(tfidf_matrix):
#     return cosine_similarity(tfidf_matrix, tfidf_matrix)

# def get_recommendations(movie_titles, similarity_matrix, df, title_col='title'):
#     sim_scores_suma = [0] * len(df)
#     for movie_title in movie_titles:
#         idx = df.index[df[title_col] == movie_title].tolist()[0]
#         sim_scores = list(enumerate(similarity_matrix[idx]))
#         for i, score in sim_scores:
#             sim_scores_suma[i] += score
    
#     numero_movies = len(movie_titles)
#     sim_scores_media = [score / numero_movies for score in sim_scores_suma]
    
#     for i in range(len(df)):
#         df.at[i, 'Puntuacion_des'] = sim_scores_media[i]
        
#     return df

# def generar_recomendaciones(dfme, dfre):
#     if (dfme['Rating'] == 5).any():
#         peliculas_5 = dfme.loc[dfme['Rating'] == 5, 'title'].tolist()
#         dfre['Puntuacion_des'] = 0
#         dfre['Puntuacion_des'] = dfre['Puntuacion_des'].astype(float)
#         if len(peliculas_5) == 1:
#             movie_title = peliculas_5[0]
#             recommendations = get_recommendations([movie_title], similarity_matrix, dfre)
#         else:
#             recommendations = get_recommendations(peliculas_5, similarity_matrix, dfre)
#         return recommendations
#     else:
#         print("No hay ninguna película con un rating de 5.")
#         return dfre
    
# dfre = preprocesar_descripciones(dfre)
# tfidf_matrix, vectorizer = vectorizar_descripciones(dfre)
# similarity_matrix = calcular_matriz_similitud(tfidf_matrix)
# recomendaciones = generar_recomendaciones(dfme, dfre)


# def filtrar_titulos(df_base, df_exclusion, columna_titulo_base='title', columna_titulo_exclusion='title'):

#     return df_base[~df_base[columna_titulo_base].isin(df_exclusion[columna_titulo_exclusion])]
# dfre = filtrar_titulos(dfre, dfme)

# def calcular_puntuacion(row):
#     if row["type"] == "MOVIE":
#         if 'Puntuacion_des' in dfre.columns and dfre['Puntuacion_des'].max() != 0:
#             return (
#                 0.65 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min())+
#                 0.12 * row["Puntuaciond"] / dfre["Puntuaciond"].max() +
#                 0.06 * row["Puntuacionac"] / dfre["Puntuacionac"].max() +
#                 0.06 * row['Puntuacion_des'] / dfre['Puntuacion_des'].max() +
#                 0.11 * (row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
#             )
#         else:
#             return (
#                 0.66 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
#                 0.15 * row["Puntuaciond"] / dfre["Puntuaciond"].max() +
#                 0.07 * row["Puntuacionac"] / dfre["Puntuacionac"].max()+
#                 0.12 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
#             )
#     else:
#         if 'Puntuacion_des' in dfre.columns and dfre['Puntuacion_des'].max() != 0:
#             return (
#                 0.67 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
#                 0.13 * row["Puntuacionac"] / dfre["Puntuacionac"].max() +
#                 0.08 * row['Puntuacion_des'] / dfre['Puntuacion_des'].max() +
#                 0.12 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
#             )
#         else:
#             return (
#                 0.7 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
#                 0.16 * row["Puntuacionac"] / dfre["Puntuacionac"].max()+
#                 0.14 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
#             )

# dfre["Puntuacion"] = dfre.apply(calcular_puntuacion, axis=1)

# def filtrar_y_ordenar_recomendaciones(dfre, columna_tipo='type', columna_puntuacion='Puntuacion', num_recomendaciones=6):
#     df_movies = dfre[dfre[columna_tipo] == "MOVIE"]
#     df_shows = dfre[dfre[columna_tipo] == "SHOW"]
    
#     recomendaciones_movies = df_movies.sort_values(by=columna_puntuacion, ascending=False).head(num_recomendaciones)
#     recomendaciones_shows = df_shows.sort_values(by=columna_puntuacion, ascending=False).head(num_recomendaciones)
    
#     return recomendaciones_movies, recomendaciones_shows

# def mostrar_recomendaciones(recomendaciones_movies, recomendaciones_shows):
#     print("Recomendaciones para películas:")
#     print(recomendaciones_movies[['title', 'Puntuacion']])
    
#     print("\nRecomendaciones para shows:")
#     print(recomendaciones_shows[['title', 'Puntuacion']])

# recomendaciones_movies, recomendaciones_shows = filtrar_y_ordenar_recomendaciones(dfre)
# mostrar_recomendaciones(recomendaciones_movies, recomendaciones_shows)
import pandas as pd
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def read_reco():
    return pd.read_csv('source/cartelera.csv')

def obtener_categorias_unicas(dfre, columna_generos='genres'):
    categorias_unicas = set()
    for genero in dfre[columna_generos].dropna().values:
        generos = genero.strip("[]").replace("'", "").replace('"', "").split(", ")
        generos = [g.strip().lower() for g in generos]
        categorias_unicas.update(generos)
    return list(categorias_unicas)

def crear_matriz_generos(dfme, categorias_unicas, columna_generos='genres', columna_rating='Rating'):
    datos = []
    for row in dfme[dfme[columna_rating] > 0][columna_generos].dropna().values:
        categorias_peliculas = []
        row_generos = row.strip("[]").replace("'", "").replace('"', "").split(", ")
        row_generos = [g.strip().lower() for g in row_generos]
        for cat in categorias_unicas:
            categorias_peliculas.append(int(cat in row_generos))
        datos.append(categorias_peliculas)
    return pd.DataFrame(data=datos, columns=list(categorias_unicas))

def crear_matriz_ponderada(dfme, df_generos_peliculas, categorias_unicas, columna_rating='Rating'):
    weighted_genre_matrix = pd.concat([dfme, df_generos_peliculas], axis=1)
    weighted_genre_matrix = (weighted_genre_matrix[categorias_unicas].values.T * weighted_genre_matrix[columna_rating].values).T
    return pd.DataFrame(weighted_genre_matrix, columns=categorias_unicas)

def calcular_pesos_usuario(weighted_genre_matrix):
    usuario_pesos = weighted_genre_matrix.sum()
    return usuario_pesos / usuario_pesos.sum()

def aplicar_penalizacion(usuario_pesos, factor_penalizacion=0.7):
    umbral_penalizacion = usuario_pesos.sort_values(ascending=False).iloc[3]
    usuario_pesos_penalizados = usuario_pesos.apply(lambda x: (x - umbral_penalizacion) * factor_penalizacion if x < umbral_penalizacion else x)
    return usuario_pesos_penalizados

def calcular_puntuacion_generos(dfre, df_generos_peliculas, usuario_pesos_penalizados, categorias_unicas):
    dfre2 = pd.concat([dfre, df_generos_peliculas], axis=1)
    dfre['Puntuacionge'] = (dfre2[categorias_unicas] * usuario_pesos_penalizados)
    return dfre

def obtener_directores_unicos(df, columna_directores='director_ids'):
    df = df.dropna(subset=[columna_directores])
    df.reset_index(drop=True, inplace=True)
    return df[columna_directores].unique().tolist()

def crear_matriz_directores(df, directores, columna_directores='director_ids'):
    datos_directores = []
    for row in df[columna_directores].values:
        c_directores = [1 if director == row else 0 for director in directores]
        datos_directores.append(c_directores)
    return pd.DataFrame(data=datos_directores, columns=directores)

def crear_matriz_ponderada_directores(df, df_directores, directores, columna_rating='Rating'):
    weighted_genre_matrix2 = pd.concat([df, df_directores], axis=1)
    weighted_genre_matrix2 = (weighted_genre_matrix2[directores].values.T * weighted_genre_matrix2[columna_rating].values).T
    return pd.DataFrame(weighted_genre_matrix2, columns=directores)

def calcular_pesos_directores(weighted_genre_matrix2, factor_penalizacion2=1.35):
    usuario_d = weighted_genre_matrix2.sum()
    usuario_d = usuario_d / usuario_d.sum()
    umbral_penalizacion2 = usuario_d.sort_values(ascending=False).iloc[1]
    usuario_dp = usuario_d.apply(lambda x: x * factor_penalizacion2 if x >= umbral_penalizacion2 else x)
    return usuario_dp

def calcular_puntuacion_directores(dfre, df_directores, usuario_dp, directores):
    dfre3 = pd.concat([dfre, df_directores], axis=1)
    dfre["Puntuaciond"] = (dfre3[directores] * usuario_dp).sum(axis=1)
    return dfre

def obtener_actores_unicos(df, columna_actores='actor_ids'):
    df[columna_actores] = df[columna_actores].fillna('').astype(str)
    actores_unicos = set()
    for actor in df[columna_actores].values:
        actores = actor.split(", ")
        actores_unicos.update(actores)
    return list(actores_unicos)

def crear_matriz_actores(df, actores_unicos, columna_actores='actor_ids', columna_rating=None):
    df[columna_actores] = df[columna_actores].fillna('').astype(str)
    datos = []
    if columna_rating:
        df_filtrado = df[df[columna_rating] > 0]
    else:
        df_filtrado = df
    
    for row in df_filtrado[columna_actores].values:
        actores_peliculas = [1 if actor in row.split(", ") else 0 for actor in actores_unicos]
        datos.append(actores_peliculas)
    return pd.DataFrame(data=datos, columns=actores_unicos)

def crear_matriz_ponderada_actores(df, df_actores_peliculas, actores_unicos, columna_rating='Rating'):
    weighted_genre_matrix = pd.concat([df, df_actores_peliculas], axis=1)
    weighted_genre_matrix = (weighted_genre_matrix[actores_unicos].values.T * weighted_genre_matrix[columna_rating].values).T
    return pd.DataFrame(weighted_genre_matrix, columns=actores_unicos)

def calcular_pesos_actores(weighted_genre_matrix):
    usuario_ac = weighted_genre_matrix.sum()
    return usuario_ac / usuario_ac.sum()

def calcular_puntuacion_actores(dfre, df_actores_peliculas, usuario_ac, actores_unicos):
    dfre4 = pd.concat([dfre, df_actores_peliculas], axis=1)
    dfre["Puntuacionac"] = (dfre4[actores_unicos] * usuario_ac).sum(axis=1)
    return dfre

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocesar_descripciones(df, columna='description'):
    df['processed_description'] = df[columna].apply(preprocess_text)
    return df

def vectorizar_descripciones(df, columna='processed_description'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[columna])
    return tfidf_matrix, vectorizer

def calcular_matriz_similitud(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_titles, similarity_matrix, df, title_col='title'):
    sim_scores_suma = [0] * len(df)
    for movie_title in movie_titles:
        idx = df.index[df[title_col] == movie_title].tolist()[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        for i, score in sim_scores:
            sim_scores_suma[i] += score

def get_recommendations(movie_titles, similarity_matrix, df, title_col='title'):
    sim_scores_suma = [0] * len(df)
    for movie_title in movie_titles:
        idx = df.index[df[title_col] == movie_title].tolist()[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        for i, score in sim_scores:
            sim_scores_suma[i] += score
    return sim_scores_suma

def generar_recomendaciones(dfme, dfre):
    tfidf_matrix, vectorizer = vectorizar_descripciones(dfre)
    similarity_matrix = calcular_matriz_similitud(tfidf_matrix)
    movie_titles = dfme['title'].tolist()
    sim_scores_suma = get_recommendations(movie_titles, similarity_matrix, dfre)
    dfre['sim_score'] = sim_scores_suma
    return dfre

def filtrar_titulos(dfre, dfme):
    titles_vistos = dfme['title'].tolist()
    return dfre[~dfre['title'].isin(titles_vistos)]

def calcular_puntuacion(row):
    return 0.4 * row['Puntuacionge'] + 0.3 * row['Puntuaciond'] + 0.3 * row['Puntuacionac']

def filtrar_y_ordenar_recomendaciones(dfre, top_n=10):
    dfre = dfre.sort_values(by='Puntuacion', ascending=False).head(top_n)
    peliculas = dfre[dfre['type'] == 'movie']
    series = dfre[dfre['type'] == 'show']
    return peliculas, series

def mostrar_recomendaciones(recomendaciones_movies, recomendaciones_shows):
    st.write("Películas recomendadas:")
    for _, row in recomendaciones_movies.iterrows():
        st.write(f"{row['title']} - Puntuación: {row['Puntuacion']:.2f}")
    
    st.write("Series recomendadas:")
    for _, row in recomendaciones_shows.iterrows():
        st.write(f"{row['title']} - Puntuación: {row['Puntuacion']:.2f}")

def calcular_y_mostrar_recomendaciones():
    dfre = read_reco()
    dfme = st.session_state['valoraciones_df']
    dfme = pd.merge(left=dfre, right=dfme, left_on='title', right_on='title', how='inner')

    categorias_unicas = obtener_categorias_unicas(dfre)
    df_generos_peliculas = crear_matriz_generos(dfme, categorias_unicas)
    weighted_genre_matrix = crear_matriz_ponderada(dfme, df_generos_peliculas, categorias_unicas)
    usuario_pesos = calcular_pesos_usuario(weighted_genre_matrix)
    usuario_pesos_penalizados = aplicar_penalizacion(usuario_pesos)
    dfre = calcular_puntuacion_generos(dfre, df_generos_peliculas, usuario_pesos_penalizados, categorias_unicas)

    directores = obtener_directores_unicos(dfme)
    df_directores_me = crear_matriz_directores(dfme, directores)
    weighted_genre_matrix2 = crear_matriz_ponderada_directores(dfme, df_directores_me, directores)
    usuario_dp = calcular_pesos_directores(weighted_genre_matrix2)
    df_directores_re = crear_matriz_directores(dfre, directores)
    dfre = calcular_puntuacion_directores(dfre, df_directores_re, usuario_dp, directores)

    actores_unicos = obtener_actores_unicos(dfme)
    df_actores_peliculas_me = crear_matriz_actores(dfme, actores_unicos, columna_rating='Rating')
    weighted_genre_matrix = crear_matriz_ponderada_actores(dfme, df_actores_peliculas_me, actores_unicos)
    usuario_ac = calcular_pesos_actores(weighted_genre_matrix)
    df_actores_peliculas_re = crear_matriz_actores(dfre, actores_unicos)
    dfre = calcular_puntuacion_actores(dfre, df_actores_peliculas_re, usuario_ac, actores_unicos)

    dfre = preprocesar_descripciones(dfre)
    tfidf_matrix, vectorizer = vectorizar_descripciones(dfre)
    similarity_matrix = calcular_matriz_similitud(tfidf_matrix)
    recomendaciones = generar_recomendaciones(dfme, dfre)

    dfre = filtrar_titulos(dfre, dfme)
    dfre["Puntuacion"] = dfre.apply(calcular_puntuacion, axis=1)
    recomendaciones_movies, recomendaciones_shows = filtrar_y_ordenar_recomendaciones(dfre)
    
    return recomendaciones_movies[['title', 'Puntuacion']].to_dict('records')






