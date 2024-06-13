import streamlit as st
import plotly.express as px
import numpy as np


def main():
    # st.set_page_config(**PAGE_CONFIG)

    menu = ["Main App", "Exploratory Data Analysis", "Machine Learning Model", "About"]

    choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)

    if choice == "Main App":
        st.subheader(body = "Home :house:")

        st.write("Welcome to the **CO2 Emissions Machine Learning Model Website** made with **Streamlit**.")

        st.markdown("""The data for this project comes from the following website: 
                       [Open Canada](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64).""")

        st.write("""To use this app just go to the `Exploratory Data Analysis` section to know more about the data that we used to build
                    the Machine Learning models.""")
        
        st.write("""To use the `Machine Learning Model` section you can either use the sliders in the sidebar or upload you own CSV file.""")

        st.warning("""Note: If you are using a CSV file you cannot use the sidebar's sliders to use the model.""") 



if __name__ == "__main__":
    main()