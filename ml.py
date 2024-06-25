import streamlit as st
import numpy as np
import pandas as pd
from module.ml_func import *
from module.recomendador_func import *



def ml_app():
    
    st.subheader(body = "RECOMENDADOR")
    st.write("En esta sección puedes comprobar como funciona el recomendador, tienes que valorar un mínimo de 5 peliculas o series con una puntación entre 1-5.")
    df = read_reco()

    if 'valoraciones_df' not in st.session_state:
        st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "Rating"])

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
            new_rating = pd.DataFrame([{"title": choice_peliculas, "Rating": int(valoracion)}])
            st.session_state['valoraciones_df'] = pd.concat([st.session_state['valoraciones_df'], new_rating], ignore_index=True)
            st.success(f"Guardada la valoración de {valoracion} para la película {choice_peliculas}")
    
    # Mostrar valoraciones guardadas
    if not st.session_state['valoraciones_df'].empty:
        st.write("Valoraciones Guardadas:")
        st.dataframe(st.session_state['valoraciones_df'])
        if st.button('Borrar valoraciones'):
            st.session_state['valoraciones_df'] = pd.DataFrame(columns=["title", "Rating"])
            st.success("Se han borrado todas las valoraciones")
    
    # Generar recomendaciones cuando hay al menos 5 valoraciones
    if len(st.session_state['valoraciones_df']) >= 5:
        if st.button('Generar recomendaciones'):
            st.write("Recomendaciones basadas en tus valoraciones:")
            df1 = st.session_state['valoraciones_df']
            dfre = df.copy()
            dfme = pd.merge(left = dfre, right= df1, left_on= 'title', right_on= 'title', how= 'inner')
        #generacion de los pesos por genero de cada usuario#
            categorias_unicas = set()
            for genero in dfre['genres'].dropna().values:
    # Limpiar las categorías eliminando corchetes, comillas y espacios innecesarios
                generos = genero.strip("[]").replace("'", "").replace('"', "").split(", ")
                generos = [g.strip().lower() for g in generos]
                categorias_unicas.update(generos)

            categorias_unicas = list(categorias_unicas)

# Crear la matriz de datos para las categorías
            datos = []

            for row in dfme[dfme["Rating"] > 0]['genres'].dropna().values:
                categorias_peliculas = []
                row_generos = row.strip("[]").replace("'", "").replace('"', "").split(", ")
                row_generos = [g.strip().lower() for g in row_generos]
                for cat in categorias_unicas:
                    categorias_peliculas.append(int(cat in row_generos))
                datos.append(categorias_peliculas)

#categorias_peliculas.append(int(cat in row.split(", ")))

            df_generos_peliculas = pd.DataFrame(data = datos, columns = list(categorias_unicas))
            dfme.reset_index(drop=True, inplace=True)

            weighted_genre_matrix = pd.concat([dfme, df_generos_peliculas], axis = 1)
            weighted_genre_matrix = (weighted_genre_matrix[categorias_unicas].values.T * weighted_genre_matrix['Rating'].values).T
            weighted_genre_matrix = pd.DataFrame(weighted_genre_matrix, columns = categorias_unicas)

            usuario_pesos = weighted_genre_matrix.sum()

            usuario_pesos = usuario_pesos / usuario_pesos.sum()

            for genero in categorias_unicas:
                if genero not in usuario_pesos:
                    usuario_pesos[genero] = 0.0000

            factor_penalizacion = 0.7
            umbral_penalizacion = usuario_pesos.sort_values(ascending=False).iloc[3] 
            usuario_pesos_penalizados = usuario_pesos.apply(lambda x: (x - umbral_penalizacion) * factor_penalizacion if x < umbral_penalizacion else x)
            datos = []
            for row in dfre['genres'].dropna().values:
                categorias_peliculas = []
                row_generos = row.strip("[]").replace("'", "").replace('"', "").split(", ")
                row_generos = [g.strip().lower() for g in row_generos]
            for cat in categorias_unicas:
                categorias_peliculas.append(int(cat in row_generos))
            datos.append(categorias_peliculas)


            df_generos_peliculas = pd.DataFrame(data=datos, columns=list(categorias_unicas))
            dfre2 = pd.concat([dfre, df_generos_peliculas], axis=1)
            dfre['Puntuacionge'] = (dfre2[categorias_unicas] * usuario_pesos_penalizados).sum(axis=1)

            dfm2=dfme.dropna(subset = ['director_ids'])
            dfm2.reset_index(drop=True, inplace=True)
            directores = dfm2['director_ids'].unique().tolist()

            datos_directores = []

            for row in dfm2[dfm2["Rating"] > 0]["director_ids"].values:
    
                c_directores = []
    
                for director in directores :
    

                    if director == row:
                        c_directores.append(1)  
                    else:
                        c_directores.append(0)  
    

                datos_directores.append(c_directores)


            df_directores = pd.DataFrame(data = datos_directores, columns = list(directores))

            weighted_genre_matrix2 = pd.concat([dfm2, df_directores], axis = 1)
            weighted_genre_matrix2 = (weighted_genre_matrix2[directores].values.T * weighted_genre_matrix2['Rating'].values).T
            weighted_genre_matrix2 = pd.DataFrame(weighted_genre_matrix2, columns = directores )
            usuario_d = weighted_genre_matrix2.sum()
            usuario_d = usuario_d / usuario_d.sum()
            factor_penalizacion2 = 1.35
            umbral_penalizacion2 = usuario_d.sort_values(ascending=False).iloc[1] 
            usuario_dp = usuario_d.apply(lambda x:x * factor_penalizacion2 if x >= umbral_penalizacion2 else x)

    # generar dataframe columnas por cada director (en los pesos del usuario) y generar puntuacion por director
            datos_directores = []

            for row in dfre['director_ids'].values:
    
                c_directores = []
    
                for director in directores :
    

                    if director == row:
                        c_directores.append(1)  
                    else:
                        c_directores.append(0)  
    

                datos_directores.append(c_directores)


            df_directores = pd.DataFrame(data=datos_directores, columns=list(directores))
            dfre3 = pd.concat([dfre, df_directores], axis=1)
            dfre["Puntuaciond"]= (dfre3[directores] * usuario_dp).sum(axis=1)


    #generacion de pesos del usuario por actor
            dfme['actor_ids'] = dfme['actor_ids'].fillna('')
            actores_unicos = set()
            for actor in dfme['actor_ids'].values:
                actores = actor.split(", ")
                actores_unicos.update(actores)

            actores_unicos = list(actores_unicos)



            datos = []

            for row in dfme[dfme["Rating"] > 0]["actor_ids"].values:

                actores_peliculas = list()
    
                for cat in actores_unicos:
    
                    if cat in row.split(", "):
                        actores_peliculas.append(1)
                    else:
                        actores_peliculas.append(0)

                datos.append(actores_peliculas)


            df_actores_peliculas = pd.DataFrame(data = datos, columns = list(actores_unicos))
            dfme.reset_index(drop=True, inplace=True)

            weighted_genre_matrix = pd.concat([dfme, df_actores_peliculas], axis = 1)
            weighted_genre_matrix = (weighted_genre_matrix[actores_unicos].values.T * weighted_genre_matrix['Rating'].values).T
            weighted_genre_matrix = pd.DataFrame(weighted_genre_matrix, columns = actores_unicos)

            usuario_ac = weighted_genre_matrix.sum()

            usuario_ac = usuario_ac / usuario_ac.sum()

            datos = []
            dfre['actor_ids'] = dfre['actor_ids'].fillna('')
            for row in dfre['actor_ids'].values:
                actores_peliculas = list()
                for cat in actores_unicos:
                    if cat in row.split(", "):
                        actores_peliculas.append(1)
                    else:
                        actores_peliculas.append(0)
                datos.append(actores_peliculas)

            df_actores_peliculas = pd.DataFrame(data=datos, columns=list(actores_unicos))
            dfre4 = pd.concat([dfre, df_actores_peliculas], axis=1)
            dfre["Puntuacionac"]=(dfre4[actores_unicos] * usuario_ac).sum(axis=1)

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            def preprocess_text(text):
                if pd.isnull(text):
                    return ""
                tokens = word_tokenize(str(text).lower())
                tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
                tokens = [word for word in tokens if word not in stop_words]
                return " ".join(tokens)
    
# Preprocesa las descripciones
            dfre5 = pd.DataFrame()
            dfre5['processed_description'] = dfre['description'].apply(preprocess_text)

# Vectoriza las descripciones preprocesadas
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(dfre5['processed_description'])

# Calcula la matriz de similitud de coseno
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define la función de recomendación
            def get_recommendations(movie_titles, similarity_matrix, df):
                sim_scores_suma = [0] * len(df)
                for movie_title in movie_titles:
                    idx = df.index[df['title'] == movie_title].tolist()[0]
                    sim_scores = list(enumerate(similarity_matrix[idx]))
                    for i, score in sim_scores:
            #if i != idx:  # Omitir la película misma
                        sim_scores_suma[i] += score
    
                numero_movies = len(movie_titles)
                sim_scores_media = [score / numero_movies for score in sim_scores_suma]
    

                for i in range(len(df)):
                    df.at[i, 'Puntuacion_des'] = sim_scores_media[i]
        
                return df


            if (dfme['Rating'] == 5).any():
                peliculas_5 = dfme.loc[dfme['Rating'] == 5, 'title'].tolist()
                if len(peliculas_5) == 1:
                    movie_title = peliculas_5[0]
                    dfre['Puntuacion_des'] = 0
                    dfre['Puntuacion_des'] = dfre['Puntuacion_des'].astype(float)
                    recommendations = get_recommendations([movie_title], similarity_matrix, dfre)
                else:
                    dfre['Puntuacion_des'] = 0
                    dfre['Puntuacion_des'] = dfre['Puntuacion_des'].astype(float)
                    recommendations = get_recommendations(peliculas_5, similarity_matrix, dfre)
            
                

            dfre = dfre[~dfre['title'].isin(dfme['title'])]
            dfre = dfre.copy()

            def calcular_puntuacion(row):
                if row["type"] == "MOVIE":
                    if 'Puntuacion_des' in dfre.columns and dfre['Puntuacion_des'].max() != 0:
                        return (
                            0.63 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min())+
                            0.10 * row["Puntuaciond"] / dfre["Puntuaciond"].max() +
                            0.06 * row["Puntuacionac"] / dfre["Puntuacionac"].max() +
                            0.10 * row['Puntuacion_des'] / dfre['Puntuacion_des'].max() +
                            0.11 * (row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
            )
                    else:
                        return (
                            0.66 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
                            0.15 * row["Puntuaciond"] / dfre["Puntuaciond"].max() +
                            0.07 * row["Puntuacionac"] / dfre["Puntuacionac"].max()+
                            0.12 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
            )
                else:
                    if 'Puntuacion_des' in dfre.columns and dfre['Puntuacion_des'].max() != 0:
                        return (
                            0.64 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
                            0.13 * row["Puntuacionac"] / dfre["Puntuacionac"].max() +
                            0.11 * row['Puntuacion_des'] / dfre['Puntuacion_des'].max() +
                            0.12 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
            )
                    else:
                        return (
                            0.7 * (row['Puntuacionge'] - dfre['Puntuacionge'].min()) / (dfre['Puntuacionge'].max() - dfre['Puntuacionge'].min()) +
                            0.16 * row["Puntuacionac"] / dfre["Puntuacionac"].max()+
                            0.14 *(row['imdb_score'] - dfre['imdb_score'].min()) / (dfre['imdb_score'].max() - dfre['imdb_score'].min())
            )

            dfre["Puntuacion"] = dfre.apply(calcular_puntuacion, axis=1)
            dfrem = dfre[dfre["type"] == "MOVIE"]
            dfres = dfre[dfre["type"] == "SHOW"]
            recomendaciones1 = dfrem.sort_values(by='Puntuacion', ascending=False)
            recomendaciones2 = dfres.sort_values(by='Puntuacion', ascending=False)

            st.write("Aquí tienes algunas recomendaciones de películas para ti:")
            st.dataframe(recomendaciones1[['title']].head(6))


            st.write("Aquí tienes algunas recomendaciones de series para ti:")
            st.dataframe(recomendaciones2[['title']].head(6))
    
        # st.write("Aquí tienes algunas recomendaciones de películas para ti:")
        # for i, rec in enumerate(recomendaciones):
        #     st.write(f"{i+1}. {rec['title']} - Puntuación: {rec['Puntuacion']}")




if __name__ == "__ml_app__":
    ml_app()


