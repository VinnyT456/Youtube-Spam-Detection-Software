#Preprocess the comment(s)
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import streamlit as st
import joblib
import contractions
import emoji
import nltk
import re

#Import the necessary libraries to train the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Models available
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import zipfile
import io

st.set_page_config(page_title="Create Your Own Model")

class YoutubeApp:
    def __init__(self):
        self.model = None
        self.current_model = ''
        self.tfidf = TfidfVectorizer(
            max_df=0.7,
            max_features=5000,
            min_df=1,
            ngram_range=(1,2),
            sublinear_tf=True,
            use_idf=False
        )
        self.tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
        self.comment = None
        self.column_labels = {}

        self.model_parameters = {
            "Logistic Regression": {
                'Penalty': ['l2', 'l1', 'elasticnet', 'none'],
                'C': [1,0.001, 0.01, 0.1, 10, 100, 1000],
                'Solver': ['lbfgs','newton-cg', 'liblinear', 'sag', 'saga'],
            },
            "Naive Bayes": {
                'Alpha': [1,0.001, 0.01, 0.1, 10, 100]
            },
            "SVC": {
                'C': [1,0.001, 0.01, 0.1, 10, 100, 1000],
                'Kernel': ['rbf','linear', 'poly', 'sigmoid'], 
                'Gamma': ['scale', 'auto'], 
            },
            "Linear SVC": {
                'C': [1,0.001, 0.01, 0.1, 10, 100, 1000],  
                'Penalty': ['l2', 'l1'],  
                'Loss': ['squared_hinge','hinge']
            },
            "Decision Tree Classifier": {
                'Criterion': ['gini', 'entropy', 'log_loss'],
                'Splitter': ['best', 'random'],
                'Max Depth': [None, 5, 10, 20, 50, 100],
                'Min Samples Split': [2, 5, 10, 20],
                'Min Samples Leaf': [1, 2, 5, 10], 
                'Max Features': ['None', 'sqrt', 'log2'],
            },
            "Random Forest Classifier": {
                'N Estimators': [100,10, 50, 200, 500], 
                'Criterion': ['gini', 'entropy', 'log_loss'],
                'Max Depth': [None, 5, 10, 20, 50, 100],
                'Min Samples Split': [2, 5, 10, 20],
                'Min Samples Leaf': [1, 2, 5, 10], 
                'Max Features': ['None', 'sqrt', 'log2'], 
            }
        }

        self.hyper_parameter = {
            'Logistic Regression':{},
            'Naive Bayes':{},
            'SVC':{},
            'Linear SVC':{},
            'Decision Tree Classifier':{},
            'Random Forest Classifier':{}
        }

    def load_logistic_regression_model(self):
        self.model = LogisticRegression(
            penalty=self.hyper_parameter['Logistic Regression'].get('penalty','l2'),
            C=self.hyper_parameter['Logistic Regression'].get('c',1),
            solver=self.hyper_parameter['Logistic Regression'].get('solver','lbfgs'),
            random_state=42, 
        )
        self.current_model = "Logistic Regression"
        st.session_state.train = False

    def load_naive_bayes_model(self):
        self.model = MultinomialNB(
            alpha=self.hyper_parameter['Naive Bayes'].get('alpha',1)
        )
        self.current_model = "Naive Bayes"
        st.session_state.train = False

    def load_SVC_model(self):
        self.model = SVC(
            C=self.hyper_parameter['SVC'].get('c',1),
            kernel=self.hyper_parameter['SVC'].get('kernel','rbf'),
            gamma=self.hyper_parameter['SVC'].get('gamma','scale'),
            random_state=42
        )
        self.current_model = "SVC"
        st.session_state.train = False

    def load_LinearSVC_model(self):
        self.model = LinearSVC(
            C=self.hyper_parameter['Linear SVC'].get('c',1),
            penalty=self.hyper_parameter['Linear SVC'].get('penalty','l2'),
            loss=self.hyper_parameter['Linear SVC'].get('loss','squared_hinge'),
            random_state=42,
        )
        self.current_model = "Linear SVC"
        st.session_state.train = False

    def load_decision_tree_model(self):
        self.model = DecisionTreeClassifier(
            criterion=self.hyper_parameter['Decision Tree Classifier'].get('criterion','gini'),
            splitter=self.hyper_parameter['Decision Tree Classifier'].get('splitter','best'),
            max_depth=self.hyper_parameter['Decision Tree Classifier'].get('max depth',None),
            min_samples_split=self.hyper_parameter['Decision Tree Classifier'].get('min samples split',2),
            min_samples_leaf=self.hyper_parameter['Decision Tree Classifier'].get('min samples leaf',1),
            max_features=self.hyper_parameter['Decision Tree Classifier'].get('max features',None),
            random_state=42,
        )
        self.current_model = "Decision Tree Classifier"
        st.session_state.train = False

    def load_random_forest_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.hyper_parameter['Random Forest Classifier'].get('n estimators', 100),
            criterion=self.hyper_parameter['Random Forest Classifier'].get('criterion', 'gini'),
            max_depth=self.hyper_parameter['Random Forest Classifier'].get('max depth', None),
            min_samples_split=self.hyper_parameter['Random Forest Classifier'].get('min samples split', 2),
            min_samples_leaf=self.hyper_parameter['Random Forest Classifier'].get('min samples leaf', 1),
            max_features=self.hyper_parameter['Random Forest Classifier'].get('max features', None),
            random_state=42
        )
        self.current_model = "Random Forest Classifier"
        st.session_state.train = False

    def select_model(self):
        st.session_state.train = False

    def is_full_width(self, text):
        """Check if the text contains full-width characters."""
        for char in text:
            if ord(char) >= 0xFF01 and ord(char) <= 0xFF5E:
                return True
        return False

    def full_width_to_half_width(self, text):
        """Converts full-width characters to half-width."""
        return "".join(
            chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text
        )

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
    
    def preprocess_batch(self, comments):
        """Process a batch of comments with a progress bar in Streamlit."""
        total_comments = len(comments)
        progress_bar = st.sidebar.progress(0)
        processed_comments = []

        for i, comment in enumerate(comments):
            processed_comments.append(self.preprocess_comment(comment))
            progress_bar.progress((i + 1) / total_comments)  # Update progress

        return processed_comments

    def fine_tune(self):
        if 'hyper_parameter' not in st.session_state:
            st.session_state['hyper_parameter'] = self.hyper_parameter
        
        # Fine-tuning the model parameters using Streamlit's selectbox
        for parameter_name, parameter_values in self.model_parameters[self.current_model].items():
            if isinstance(parameter_values, list):
                selected_value = st.sidebar.selectbox(f"Select {parameter_name}", options=parameter_values)
                if selected_value != 'Select':
                    st.session_state['hyper_parameter'][self.current_model][parameter_name.lower()] = selected_value

    def train_model(self):
        comment = st.session_state.feature_col
        label = st.session_state.label_col
        if comment == label:
            st.warning("Please select two different labels")
        elif self.model is None:
            st.warning("Please select the model to train")
        else:
            # Prepare the data
            df = st.session_state.df
            df = df[[comment, label]]
            df = df.drop_duplicates(subset=comment)
            df[comment] = self.preprocess_batch(df[comment])
            df.replace(np.nan, '', inplace=True)
            df = df.dropna()

            X = df[comment]
            y = df[label]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Transform text data into numerical data using TF-IDF
            X_train = self.tfidf.fit_transform(X_train)
            X_test = self.tfidf.transform(X_test)

            try:
                # Check if the model has the correct parameters and update the model based on hyperparameters
                if self.current_model == "Logistic Regression":
                    self.model = LogisticRegression(
                        penalty=st.session_state['hyper_parameter']['Logistic Regression'].get('penalty', 'l2'),
                        C=st.session_state['hyper_parameter']['Logistic Regression'].get('c', 1),
                        solver=st.session_state['hyper_parameter']['Logistic Regression'].get('solver', 'lbfgs'),
                        random_state=42
                    )
                elif self.current_model == "Naive Bayes":
                    self.model = MultinomialNB(
                        alpha=st.session_state['hyper_parameter']['Naive Bayes'].get('alpha', 1)
                    )
                elif self.current_model == "SVC":
                    self.model = SVC(
                        C=st.session_state['hyper_parameter']['SVC'].get('c', 1),
                        kernel=st.session_state['hyper_parameter']['SVC'].get('kernel', 'rbf'),
                        gamma=st.session_state['hyper_parameter']['SVC'].get('gamma', 'scale'),
                        random_state=42
                    )
                elif self.current_model == "Linear SVC":
                    self.model = LinearSVC(
                        C=st.session_state['hyper_parameter']['Linear SVC'].get('c', 1),
                        penalty=st.session_state['hyper_parameter']['Linear SVC'].get('penalty', 'l2'),
                        loss=st.session_state['hyper_parameter']['Linear SVC'].get('loss', 'squared_hinge'),
                        random_state=42
                    )
                elif self.current_model == "Decision Tree Classifier":
                    self.model = DecisionTreeClassifier(
                        criterion=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('criterion', 'gini'),
                        splitter=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('splitter', 'best'),
                        max_depth=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('max depth', None),
                        min_samples_split=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('min samples split', 2),
                        min_samples_leaf=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('min samples leaf', 1),
                        max_features=st.session_state['hyper_parameter']['Decision Tree Classifier'].get('max features', None),
                        random_state=42
                    )
                elif self.current_model == "Random Forest Classifier":
                    self.model = RandomForestClassifier(
                        n_estimators=st.session_state['hyper_parameter']['Random Forest Classifier'].get('n estimators', 100),
                        criterion=st.session_state['hyper_parameter']['Random Forest Classifier'].get('criterion', 'gini'),
                        max_depth=st.session_state['hyper_parameter']['Random Forest Classifier'].get('max depth', None),
                        min_samples_split=st.session_state['hyper_parameter']['Random Forest Classifier'].get('min samples split', 2),
                        min_samples_leaf=st.session_state['hyper_parameter']['Random Forest Classifier'].get('min samples leaf', 1),
                        max_features=st.session_state['hyper_parameter']['Random Forest Classifier'].get('max features', None),
                        random_state=42
                    )

            
                # Train the model
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                st.session_state['y'] = [y_test,y_pred]

                self.download_model()

            except:
                st.warning(f"Parameter doesn't work for {self.current_model}")

    def download_model(self):
        # Step 4: Save the model and the TF-IDF vectorizer
        model_filename = f"{self.current_model}.joblib"
        tfidf_filename = f"tfidf_vectorizer_{self.current_model}.joblib"

        # Save model and TF-IDF vectorizer
        joblib.dump(self.model, model_filename)
        joblib.dump(self.tfidf, tfidf_filename)

        # Step 5: Create a ZIP file containing both the model and the TF-IDF vectorizer
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(model_filename, arcname=model_filename)
            zip_file.write(tfidf_filename, arcname=tfidf_filename)
        
        # Seek to the start of the BytesIO buffer
        zip_buffer.seek(0)

        # Download button for the ZIP file
        st.download_button(
            label=f"Download {self.current_model} model and TF-IDF Vectorizer",
            data=zip_buffer,
            file_name=f"{self.current_model}_and_tfidf.zip",
            mime="application/zip",
            key='model_and_tfidf_download'
        )

        # Optional success message
        st.success("The model and TF-IDF vectorizer have been trained! Click the button above to download them.")

    def display_model_results(self):
        # Classification Report
        report = classification_report(st.session_state['y'][0], st.session_state['y'][1])

        # Confusion Matrix
        cm = confusion_matrix(st.session_state['y'][0], st.session_state['y'][1])

        # Display classification report
        st.subheader('Classification Report')
        st.text(report)
        
        st.markdown("<h2 style='text-align: center;'>Confusion Matrix</h2>", unsafe_allow_html=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm", cbar=False)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(plt)

    def run(self):
        page_names_to_funcs = {
            "Select Model": self.select_model,
            "Logistic Regression": self.load_logistic_regression_model,
            "Naive Bayes": self.load_naive_bayes_model,
            "SVC": self.load_SVC_model,
            "LinearSVC": self.load_LinearSVC_model,
            "Decision Tree": self.load_decision_tree_model,
            "Random Forest": self.load_random_forest_model,
        }

        demo_name = st.sidebar.selectbox("Choose a model", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()

        if ('feature_col' not in st.session_state):
            st.session_state.feature_col = 'Select a column' 

        if ('label_col' not in st.session_state):
            st.session_state.label_col = 'Select a column' 

        if ('disabled' not in st.session_state):
            st.session_state.disabled = False

        if ('train' not in st.session_state):
            st.session_state.train = False

        if self.model is not None:  # Model has been selected
            if (st.sidebar.button('Upload New Dataset')):
                st.session_state.disabled = False
                st.session_state.train = False
                st.session_state.feature_col = 'Select a column' 
                st.session_state.label_col = 'Select a column' 

            if st.session_state.feature_col != 'Select a column' and st.session_state.label_col != 'Select a column':
                # Allow training only when both columns are selected
                if st.sidebar.button('Train Model'):
                    st.session_state.train = True
                    st.session_state.disabled = True
                    self.train_model()

                self.fine_tune()

            if st.session_state.train:
                self.display_model_results()

            if (not st.session_state.disabled):
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

                if uploaded_file is not None:
                    # Read the CSV file into a DataFrame
                    if ('df' not in st.session_state):
                        st.session_state.df = pd.read_csv(uploaded_file)

                    # Create selection boxes for feature and label columns
                    columns = ['Select a column'] + st.session_state.df.columns.tolist()

                    # Update the selectbox for the feature column
                    feature_col_value = st.selectbox(
                        "Select the feature column (X):", 
                        columns, 
                        index=columns.index(st.session_state.feature_col)
                    )

                    # Update the session state only if the feature column is updated
                    if feature_col_value != st.session_state.feature_col:
                        st.session_state.feature_col = feature_col_value
                        st.rerun()  # Force a rerun to apply the change instantly

                    if (feature_col_value != 'Select a column'):
                        st.write(st.session_state.df[feature_col_value].head())

                    # Update the selectbox for the label column
                    label_col_value = st.selectbox(
                        "Select the label column (y):", 
                        columns, 
                        index=columns.index(st.session_state.label_col)
                    )

                    # Update the session state only if the label column is updated
                    if label_col_value != st.session_state.label_col:
                        st.session_state.label_col = label_col_value
                        st.rerun()  # Force a rerun to apply the change instantly

                    if (label_col_value != 'Select a column'):
                        st.write(st.session_state.df[label_col_value].head())

        else:
            # If no model is selected, show a warning
            st.warning("Please select a model first.")

YoutubeApp().run()
