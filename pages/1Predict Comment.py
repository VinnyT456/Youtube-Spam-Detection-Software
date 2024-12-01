import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import contractions
import emoji
import re

st.set_page_config(page_title="Visualization")

class YoutubeApp:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
        self.comment = None

    def load_logistic_regression_model(self):
        self.model = joblib.load('models/logistic_regression_model.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_logistic.joblib')

    def load_naive_bayes_model(self):
        self.model = joblib.load('models/NBModel.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_NB.joblib')

    def load_SVC_model(self):
        self.model = joblib.load('models/SVC.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_SVC.joblib')

    def load_LinearSVC_model(self):
        self.model = joblib.load('models/LinearSVC.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_LinearSVC.joblib')

    def load_decision_tree_model(self):
        self.model = joblib.load('models/DecisionTree.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_DecisionTree.joblib')

    def load_random_forest_model(self):
        self.model = joblib.load('models/RandomForest.joblib')
        self.tfidf = joblib.load('models/tfidf_vectorizer_RandomForest.joblib')

    def is_full_width(self,text):
        # Check if any character in the text is a full-width character
        for char in text:
            if ord(char) >= 0xFF01 and ord(char) <= 0xFF5E:
                return True
        return False

    def full_width_to_half_width(self,text):
        """Converts full-width characters to half-width."""
        return "".join(
            chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text
        )

    #Steps for preprocessing the comment
    def preprocess_comment(self, comment: str) -> str:
        """Preprocess the comment by performing several text cleaning steps."""
        # Convert full-width characters to half-width if necessary
        if self.is_full_width(comment):
            comment = self.full_width_to_half_width(comment)

        #Change comment to lowercase
        lowercase_comment = comment.lower()

        #Replace websites or emails with url
        url_replaced_comment = re.sub(r'http\S+|www\S+|@\S+','url',lowercase_comment)

        #Change the emojis with the word
        emoji_replaced_comment = emoji.demojize(url_replaced_comment)

        #Remove the special characters
        special_char_removed_comment = re.sub(r'[^\w\s,]', '', emoji_replaced_comment)
        special_char_removed_comment = re.sub(r',',' ',special_char_removed_comment)

        #Remove the numbers in the comment
        number_removed_comment = re.sub(r'\d+', '', special_char_removed_comment) 

        #Remove the excess space
        space_removed_comment = number_removed_comment.strip()

        # Tokenize the comment, remove stopwords, and expand contractions
        tokenized_comment = [
            contractions.fix(word) for word in self.tokenizer.tokenize(space_removed_comment)
            if word not in self.stop_words
        ]

        # Correct misspelled words using SymSpell
        corrected_comment = [
            self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2) if (word != 'url') else word# Find closest match
            for word in tokenized_comment 
        ]

        # Flatten the result of lookup (since it returns a list of possible corrections)
        corrected_comment = [
            suggestion[0].term if suggestion and (word != 'url') else word
            for word, suggestion in zip(tokenized_comment, corrected_comment)
        ]

        # If there were corrections, apply lemmatization to the corrected words
        lemmatized_comment = [self.lemmatizer.lemmatize(word) for word in corrected_comment]

        # Join the lemmatized words back into a string
        return " ".join(lemmatized_comment)

    def grab_comment(self):
        st.title("Predict Comment")
        self.comment = st.text_input("Enter your comment")
        if self.comment: 
            self.predict_comment()
        else:
            st.write("Awaiting input...")

    def grab_multiple_comments(self):
        st.title("Predict Comment")

        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if (uploaded_file is not None):
            df = pd.read_csv(uploaded_file)

            df.dropna(axis=0, inplace=True)

            columns = ['Select a column'] + df.columns.tolist()

            feature_col_value = st.selectbox(
                        "Select the column with the comments:", 
                        columns, 
            )

            if (feature_col_value != 'Select a column'):
                comments = df[feature_col_value].apply(self.preprocess_comment)
                comments = self.tfidf.transform(comments)
                predictions = self.model.predict(comments)
                df['Prediction'] = predictions
                df['Prediction'] = df['Prediction'].apply(lambda x: "Spam" if x == 1 else "Not Spam")

                st.dataframe(df[[feature_col_value, 'Prediction']])

    def predict_comment(self):
        if (self.comment is not None or self.comment != ""):
            preprocessed_comment = self.preprocess_comment(self.comment)
            tfidf_comment = self.tfidf.transform([preprocessed_comment])
            prediction = self.model.predict(tfidf_comment)
            if (prediction[0] == 1):
                st.write("This comment is spam")
            else:
                st.write("This comment is not spam")

    def run(self):
        page_names_to_funcs = {
            "Logistic Regression":self.load_logistic_regression_model,
            "Naive Bayes": self.load_naive_bayes_model,
            "SVM": self.load_SVC_model,
            "LinearSVC": self.load_LinearSVC_model,
            "Decision Tree": self.load_decision_tree_model,
            "Random Forest": self.load_random_forest_model,
        }

        demo_name = st.sidebar.selectbox("Choose a model", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()

        if ('single_comment_prediction' not in st.session_state):
            st.session_state.single_comment_prediction = False

        if ('multiple_comment_prediction' not in st.session_state):
            st.session_state.multiple_comment_prediction = False

        if (st.sidebar.button('Single Comment Prediction')):
            st.session_state.single_comment_prediction = True
            st.session_state.multiple_comment_prediction = False

        if (st.sidebar.button('Multiple Comment Prediction')):
            st.session_state.multiple_comment_prediction = True
            st.session_state.single_comment_prediction = False

        if (st.session_state.single_comment_prediction):
            self.grab_comment()

        if (st.session_state.multiple_comment_prediction):
            self.grab_multiple_comments()


YoutubeApp().run()
