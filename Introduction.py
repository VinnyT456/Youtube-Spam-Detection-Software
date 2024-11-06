import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome to our Youtube Spam Detection App! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This app was designed for our CS 250 final project we decided to tackle the problem of Youtube Spam Detection.
    **👈 Move to the other pages from the sidebar** to see what our app can do!
    ### Want to learn more?
    - Source code [Github](https://github.com/VinnyT456/Youtube-Spam-Detection-Software)
    - Check out the dataset we used [Dataset](https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset/data)
"""
)
