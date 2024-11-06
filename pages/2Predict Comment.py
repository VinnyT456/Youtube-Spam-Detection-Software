import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Visualization", page_icon="📈")

class YoutubeApp:
    def __init__(self):
        self.model = None
        self.tfidf = None

    def load_logistic_regression_model(self):
        self.model = joblib.load('logistic_regression_model.joblib')
        self.tfidf = joblib.load('tfidf_vectorizer_logistic.joblib')

    def load_naive_bayes_model(self):
        self.model = joblib.load('NBModel.joblib')
        self.tfidf = joblib.load('tfidf_vectorizer_NB.joblib')

    def load_SVC_model(self):
        self.model = joblib.load('SVC.joblib')
        self.tfidf = joblib.load('tfidf_vectorizer_SVC.joblib')

    def load_LinearSVC_model(self):
        self.model = joblib.load('LinearSVC.joblib')
        self.tfidf = joblib.load('tfidf_vectorizer_LinearSVC.joblib')

    def info_page(self):
        pass

    def run(self):
        page_names_to_funcs = {
            "Info":self.info_page,
            "Logistic Regression":self.load_logistic_regression_model,
            "Naive Bayes": self.load_naive_bayes_model,
            "SVM": self.load_SVC_model,
            "LinearSVC": self.load_LinearSVC_model,
        }

        demo_name = st.sidebar.selectbox("Choose a model", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()


YoutubeApp().run()