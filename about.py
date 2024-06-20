import streamlit as st


def about_app():

    st.header("SOBRE NOSOTROS")
    
    st.subheader("MIEMBROS DEL EQUIPO")

    st.text("   ")

    st.write("Los miembros del equipo que han participado en este proyecto: ")

    st.markdown("Álvaro Marcos Martín: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning| Matemático.")

    st.markdown("Igor Ayestarán García: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning.") 
    
    st.markdown("Raquel García Tajes: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning | Docente en Educación Primaria.")
    
    st.markdown("Ignacio Landín García: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning |Contable.")

    

if __name__ == "__about_app__":
    about_app()