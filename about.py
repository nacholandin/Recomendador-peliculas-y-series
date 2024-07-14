import streamlit as st


def about_app():

    st.header("SOBRE NOSOTROS")
    
    st.subheader("MIEMBROS DEL EQUIPO")

    st.text("   ")

    st.write("Los miembros del equipo que han participado en este proyecto: ")
# Alvaro Marcos
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Álvaro Marcos Martín: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning | Matemático.")
    # with col2:
        # if st.button('LINKEDIN', key='button_1'):
            # st.markdown('<a href="https://www.linkedin.com/in/ignacio-landin-garcia" target="_blank"><button>Perfil Ignacio</button></a>', unsafe_allow_html=True)

# Igor Ayestarán García
    col3, col4 = st.columns([3, 1])
    with col3:
        st.markdown("Igor Ayestarán García: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning.")
    with col4:
        if st.button('LINKEDIN', key='button_2'):
            st.markdown('<a href="https://www.linkedin.com/in/igorayestarangarcia/" target="_blank"><button>Perfil Igor</button></a>', unsafe_allow_html=True)

# Raquel García Tajes
    col5, col6 = st.columns([3, 1])
    with col5:
        st.markdown("Raquel García Tajes: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning | Docente en Educación Primaria.")
    with col6:
        if st.button('LINKEDIN', key='button_3'):
            st.markdown('<a href="https://www.linkedin.com/in/raquelgarc%C3%ADatajes/" target="_blank"><button>Perfil Raquel</button></a>', unsafe_allow_html=True)

# Ignacio Landín García
    col7, col8 = st.columns([3, 1])
    with col7:
        st.markdown("Ignacio Landín García: Junior Data Scientist | Especialista en Inteligencia Artificial, Python, SQL y Machine Learning | Contable.")
    with col8:
        if st.button('LINKEDIN', key='button_4'):
            #st.write('<meta http-equiv="refresh" content="0; url=https://www.linkedin.com/in/ignacio-landin-garcia">', unsafe_allow_html=True)
            st.markdown('<a href="https://www.linkedin.com/in/ignacio-landin-garcia" target="_blank"><button>Perfil Ignacio</button></a>', unsafe_allow_html=True)

if __name__ == "__about_app__":
    about_app()