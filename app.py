import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
import os
from module.ml_func import *
from eda import eda_app
from ml import ml_app
from about import about_app

def main():
    st.set_page_config(**PAGE_CONFIG)

    menu = ["Main App", "Exploratory Data Analysis", "Machine Learning Model", "About"]

    choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)

    if choice == "Main App":
        st.subheader(body = "Home :house:")

        st.write(".")

        st.markdown("""The data for this project comes from the following website: 
                       """)

        st.write("""To use this app just go to the `Exploratory Data Analysis` section to know more about the data that we used to build
                    the Machine Learning models.""")
        
        st.write("""To use the `Machine Learning Model` section you can either use the sliders in the sidebar or upload you own CSV file.""")

        st.warning("""Note: If you are using a CSV file you cannot use the sidebar's sliders to use the model.""") 
        image1 = Image.open("source/Neflix.jpg")
        image2 = Image.open("source/hbo.jpg")
        image3 = Image.open("source/disney-tile-gradient_007bce85.jpeg")
        image4 = Image.open("source/amazonprime.png")
        image5 = Image.open("source/paramount.png")
        images = [image1, image2, image3, image4, image5]
        num_columns = 3
        cols = st.columns(num_columns)
        for idx, image in enumerate(images):
            cols[idx % num_columns].image(image, use_column_width=True)
        #st.image(image           = image,
        #caption          = "Neflix",
         #use_column_width = True)

       
    elif choice == "Exploratory Data Analysis":
        eda_app()

    elif choice == "Machine Learning Model":
        ml_app()

    else:
        about_app()

if __name__ == "__main__":
    main()