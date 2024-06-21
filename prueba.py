#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import threading


# In[2]:


dfre = pd.read_csv('cartelera.csv')


# In[ ]:





# In[ ]:





# In[3]:


gui_finished_event = threading.Event()

def obtener_valoraciones():
    valoraciones = {}
    while True:
        # Crear una ventana de diálogo personalizada con autocompletar
        dialog = AutocompleteDialog(root, dfre['title'].values)
        root.wait_window(dialog.top)
        
        title = dialog.result
        if not title:
            break
        # Verificar si el título está en la cartelera
        if title not in dfre['title'].values:
            messagebox.showerror("Error", f"La película '{title}' no está en nuestra cartelera.")
            continue
        # Preguntar al usuario por la valoración de la película
        rating = simpledialog.askfloat("Input", f"Introduce la valoración para '{title}':")
        if rating is not None:
            valoraciones[title] = rating
    return valoraciones

def mostrar_resultado(df):
    if df.empty:
        messagebox.showinfo("Información", "No hay datos de películas valoradas para mostrar.")
    else:
        result_window = tk.Toplevel(root)
        result_window.title("Películas Valoradas")
        text = tk.Text(result_window)
        text.pack(expand=True, fill='both')
        text.insert(tk.END, df.to_string(index=False))

def main():
    global df_valoraciones_completas
    valoraciones = obtener_valoraciones()
    print("Valoraciones obtenidas:", valoraciones)  # Mensaje de diagnóstico
    if valoraciones:
        # Convertir las valoraciones a un DataFrame
        df_nuevas_valoraciones = pd.DataFrame(list(valoraciones.items()), columns=['title', 'Rating'])
        print("DataFrame de nuevas valoraciones creado:\n", df_nuevas_valoraciones)  # Mensaje de diagnóstico
        # Unir con dfre para obtener todos los datos de las películas valoradas
        df_valoraciones_completas = pd.merge(left=df_nuevas_valoraciones, right=dfre, left_on='title', right_on='title', how='inner')
        print("DataFrame de valoraciones completas creado:\n", df_valoraciones_completas)  # Mensaje de diagnóstico
        mostrar_resultado(df_valoraciones_completas)
    else:
        messagebox.showinfo("Información", "No se introdujeron valoraciones.")
    # Señalar que la GUI ha terminado
    gui_finished_event.set()

def run_gui():
    global root
    root = tk.Tk()
    root.withdraw()  # Esconder la ventana principal
    main()
    root.mainloop()

class AutocompleteDialog:
    def __init__(self, parent, suggestions):
        top = self.top = tk.Toplevel(parent)
        self.top.title("Introduce el título de la película")
        self.suggestions = suggestions
        self.result = None
        
        self.label = tk.Label(top, text="Introduce el título de la película (o deja vacío para terminar):")
        self.label.pack(side="top", fill="x", padx=20, pady=5)
        
        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(top, textvariable=self.entry_var)
        self.entry.pack(side="top", fill="x", padx=20, pady=5)
        
        self.listbox = tk.Listbox(top)
        self.listbox.pack(side="top", fill="both", expand=True, padx=20, pady=5)
        
        self.entry_var.trace("w", self.update_listbox)
        self.entry.bind("<Return>", self.on_select)
        self.listbox.bind("<Double-Button-1>", self.on_select)
        
        self.entry.focus_set()

    def update_listbox(self, *args):
        search_term = self.entry_var.get()
        self.listbox.delete(0, tk.END)
        for item in self.suggestions:
            if search_term.lower() in item.lower():
                self.listbox.insert(tk.END, item)
    
    def on_select(self, event=None):
        try:
            index = self.listbox.curselection()[0]
            self.result = self.listbox.get(index)
        except IndexError:
            self.result = self.entry_var.get()
        self.top.destroy()

# Ejecutar la interfaz gráfica en un hilo separado
thread = threading.Thread(target=run_gui)
thread.start()

# Esperar hasta que la GUI termine
gui_finished_event.wait()

# Aquí puedes continuar con otras celdas del notebook
print("La interfaz gráfica ha terminado. Continuando con las siguientes celdas del notebook.")

# Mostrar el DataFrame de valoraciones completas para asegurarnos de que está disponible
if 'df_valoraciones_completas' in globals():
    print(df_valoraciones_completas.head())
else:
    print("No hay valoraciones completas disponibles.")


# In[4]:


dfme = df_valoraciones_completas


# In[ ]:





# In[5]:


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


usuario_pesos


# In[6]:


# Definir un umbral de penalización o un factor de penalización
factor_penalizacion = 0.7
umbral_penalizacion = usuario_pesos.sort_values(ascending=False).iloc[3] 

usuario_pesos_penalizados = usuario_pesos.apply(lambda x: (x - umbral_penalizacion) * factor_penalizacion if x < umbral_penalizacion else x)
umbral_penalizacion


# In[7]:


usuario_pesos_penalizados


# In[8]:


#creo dataframe columnas por cada genero (en los pesos del usuario) y genero puntuacion por genero
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


# In[9]:


#esto es el codigo que me dijo dmitry lo dejo aqui comentado para luego intentar sustituirlo
#categorias_peliculas.append(int(cat in row.split(", ")))


# In[10]:


#generacion de pesos del usuario por director
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

usuario_d


# In[11]:


factor_penalizacion2 = 1.35
umbral_penalizacion2 = usuario_d.sort_values(ascending=False).iloc[1] 
usuario_dp = usuario_d.apply(lambda x:x * factor_penalizacion2 if x >= umbral_penalizacion2 else x)


# In[12]:


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


# In[13]:


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

usuario_ac


# In[14]:


# generar dataframe columnas por cada actor (en los pesos del usuario) y generar puntuacion por actor
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


# In[15]:


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


# In[ ]:





# In[16]:


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
else:
    print("No hay ninguna película con un rating de 5.")


# In[ ]:





# In[17]:


dfre = dfre[~dfre['title'].isin(dfme['title'])]
dfre = dfre.copy()


# Generacion de recomendaciones
# 

# In[18]:


# se normalizan las puntuaciones y se le da un peso de 65 % a genero 20 % a director, 10 a actores y 5 a descripcion
# en caso de no haya valorado ninguna pelicula con un 5 - son 65 % genero 25 % director y 10 % actores


# In[19]:


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


# In[ ]:





# In[20]:


dfrem = dfre[dfre["type"] == "MOVIE"]
dfres = dfre[dfre["type"] == "SHOW"]


# In[21]:


recomendaciones1 = dfrem.sort_values(by='Puntuacion', ascending=False)
recomendaciones2 = dfres.sort_values(by='Puntuacion', ascending=False)


# In[22]:


recomendaciones1.head(12)


# In[23]:


recomendaciones2.head(12)


# In[ ]:




