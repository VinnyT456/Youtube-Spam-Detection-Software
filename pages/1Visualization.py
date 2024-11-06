import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="Visualization", page_icon="📈")

class YoutubeApp:
    def __init__(self):
        # Initialize the page config and URL for the dataset
        self.url = 'https://raw.githubusercontent.com/VinnyT456/CS-250/refs/heads/main/Youtube-Spam-Dataset.csv'

    def load_data(self):
        # Load data from the URL
        self.data = pd.read_csv(self.url)

    def prepare_data_visualization(self):
        # Drop unnecessary column and prepare the graph data
        self.graph = self.data.drop(['DATE'], axis=1)
        self.graph['LENGTH'] = self.graph['CONTENT'].apply(lambda x: len(x))
        self.graph['CLASS'] = self.graph['CLASS'].apply(lambda x: "SPAM" if x == 1 else 'NOT SPAM')

    def plot_distribution(self):
        # Create a countplot for Spam Distribution
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x=self.graph['CLASS'], ax=self.ax)
        self.ax.set_title('Spam Distribution')
        st.pyplot(self.fig)

    def plot_box(self):
        # Create boxplots for author name length and comment length by class
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 10))

        self.graph['AUTHOR_LENGTH'] = self.graph['AUTHOR'].apply(lambda x: len(x))
        self.ax[0].set_title('Author Name Length by Class')
        sns.boxplot(y=self.graph['AUTHOR_LENGTH'], x=self.graph['CLASS'], ax=self.ax[0])

        self.graph['LENGTH'] = self.graph['CONTENT'].apply(lambda x: len(x))
        self.ax[1].set_title('Comment Length by Class')
        sns.boxplot(y=self.graph['LENGTH'], x=self.graph['CLASS'], ax=self.ax[1])

        st.pyplot(self.fig)

    def average_comment_length_plot(self):
        df = self.graph.groupby(['CLASS'])["LENGTH"].mean().reset_index(name='Average Length')

        fig = px.bar(df,
                    x="CLASS",
                    y='Average Length',
                    barmode='group',
                    title='Average Comment Lengths by Spam Classification',
                    labels={"CLASS": "Spam Classification"},
                    color_continuous_scale='picnic',
                    height=500,  # Set height
                    width=1000)  # Set width
        st.plotly_chart(fig, use_container_width=True)

    def spam_comment_count_plot(self):
        df = self.graph.groupby(['VIDEO_NAME', 'CLASS']).size().reset_index(name='count')

        fig = px.bar(df,
                    x="VIDEO_NAME",
                    y='count',
                    color="CLASS",
                    barmode='group',
                    title='Spam Comment Counts by Video',
                    labels={"VIDEO_NAME": "Video Name", "count": "Count"},
                    color_continuous_scale='picnic',
                    height=500,  # Set height
                    width=1000)  # Set width
        st.plotly_chart(fig, use_container_width=True)

    def plot_comment_length_by_video(self):
        df = self.graph.groupby(['VIDEO_NAME', 'CLASS'])["LENGTH"].mean().reset_index(name='Average Length')

        fig = px.bar(df,
                    x="VIDEO_NAME",
                    y='Average Length',
                    color="CLASS",
                    barmode='group',
                    title='Average Comment Lengths by Video and Spam Classification',
                    labels={"VIDEO_NAME": "Video Name"},
                    color_continuous_scale='picnic',
                    height=500,  # Set height
                    width=1000)  # Set width
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        # Load data and prepare data for visualization
        self.load_data()
        self.prepare_data_visualization()

        page_names_to_funcs = {
            "Spam Distribution Plot": self.plot_distribution,
            "Box Plot": self.plot_box,
            "Average Comment Length Plot": self.average_comment_length_plot,
            "Spam Comment Plot": self.spam_comment_count_plot,
        }

        demo_name = st.sidebar.selectbox("Choose a visualization", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()


YoutubeApp().run()