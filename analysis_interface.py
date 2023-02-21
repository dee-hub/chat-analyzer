import pandas as pd
import re
import pandas as pd
import plotly.express as px
import plotly.express as px
import spacy
import numpy as np
import gensim.corpora as corpora
import gensim
import pyLDAvis.gensim_models
import pyLDAvis
import os
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
#nltk.download('vader_lexicon')
#import pycountry
import re
import string
#from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import meta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#@st.experimental_memo
#peterobi = pd.read_csv('peterobi.csv')
#atiku = pd.read_csv('atiku.csv')
#bat = pd.read_csv('BAT.csv')
import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
#import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

st.set_page_config(
    page_title="WhatsApp Analyzer: Deep Exploratory Chat Analysis",
    #page_icon="logo.png",
    layout="wide",
)

# dashboard title
st.title('WhatsApp Analyzer')
st.markdown("Deep Exploratory Chat Analysis")
#st.markdown(meta.SUB_HEADER, unsafe_allow_html=True)
col1, col2= st.columns([4, 6])
# top-level filters
with col1:
    uploaded_file = st.file_uploader("Uploaded your exported WhatsApp Chat File", key="files")
    click = st.button('Analyse')
    def import_chat(file):
    # Read the file
        #with open(file, 'r', encoding='utf-8') as f:
        lines = file.readlines()

        # Extract the data using regular expressions
        data = []
        for line in lines:
            match = re.findall('\[(.*?)\] (.*?): (.*)', line.strip())
            if match:
                data.append(match[0])
        df = pd.DataFrame(data, columns=['DateTime', 'Author', 'Message'])
        df.drop([0, 1, 2], axis='index', inplace=True)
        # Create a new dataframe with only valid rows
        valid_rows = []
        for i, row in df.iterrows():
            try:
                date_time = pd.to_datetime(row['DateTime'], format='%m/%d/%y, %I:%M:%S')
                row['Date'] = date_time.date()
                row['Time'] = date_time.time()
                valid_rows.append(row)
            except ValueError as e:
                print(f"Error on row {i}: {e}")
        df = pd.DataFrame(valid_rows)
        df.reset_index(inplace=True, drop=True)
        return df

with col2:
# creating a single-element container
    if click:
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            data = import_chat(bytes_data)
            st.write(data)
            data['Author'] = data['Author'].str.replace('[^\w\s]+|\s+', '', regex=True)
            data['Author'] = data['Author'].astype(str)
            radio_list = data['Author'].unique()
            add_selectbox = st.sidebar.radio("Select a chat member to analyze", tuple(radio_list))

def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


#*********************************************

def plot_message(file):
    data = import_chat(file)
# Get the value counts for each author
    author_freq = data['Author'].value_counts()
# Get the top 10 authors by message count
    top_authors = author_freq.iloc[:10]
# Create a dataframe with the author and message count data
    data_messages = pd.DataFrame({'Author': top_authors.index, 'Message Count': top_authors.values})
# Create the plot using plotly.express
    fig = px.bar(data_messages, x='Message Count', y='Author', orientation='h',
                 labels={'Message Count': 'Message Count', 'Author': 'Author'},
                 title='Top 10 Authors by Message Count')
    fig.show()

def character_count(data):
    data = import_chat(file)
    data['Message Character Count'] = data['Message'].apply(lambda x: len(x))
    dataframe = data.copy()
# Calculate the total character count for each author
    author_character_counts = data.groupby('Author')['Message Character Count'].sum()
# Sort the authors by their total character count in descending order
    sorted_authors = author_character_counts.sort_values(ascending=False)
    sorted_authors = sorted_authors.iloc[:10]
# Create a dataframe with the author and message count data
    data_character = pd.DataFrame({'Author': sorted_authors.index, 'Character Count': sorted_authors.values})
# Create the plot using plotly.express
    fig = px.bar(data_character, x='Character Count', y='Author', orientation='h',
                 labels={'Character Count': 'Character Count', 'Author': 'Author'},
                 title='Top 10 Authors by Message Count')
    fig.show()
    
    return dataframe

def pie_chart(data):
# Calculate the total character count for each author
    data = character_count(data)
    author_character_counts = data.groupby('Author')['Message Character Count'].sum()
# Sort the authors by their total character count in descending order
    sorted_authors = author_character_counts.sort_values(ascending=False)
    top_authors = sorted_authors.iloc[:10]
# Calculate the sum of the character counts
    total_character_count = top_authors.sum()
# Calculate the percentage of character count for each author
    author_percentages = [(count / total_character_count) * 100 for count in top_authors]
# Calculate the total character count of the authors outside the top 10
    other_authors = sorted_authors.iloc[10:]
    other_character_count = other_authors.sum()
# Calculate the percentage of character count for the other authors
    other_percentage = (other_character_count / total_character_count) * 100
# Create a dataframe with the author and percentage data, including 'others'
    data = pd.DataFrame({'Author': list(top_authors.index) + ['Others'], 'Percentage': list(author_percentages) + [other_percentage]})
# Create the plot using plotly.express
    fig = px.pie(data, values='Percentage', names='Author',
                 title='Top 10 Authors by Character Count (including Others)')
    fig.show()

def line_by_hour(data):
    data = character_count(data)
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%y, %I:%M:%S %p')
# Group the messages by hour
    hourly_counts = data.groupby(data['DateTime'].dt.hour)['Message'].count().reset_index(name='Message Count')
    hourly_counts['DateTime'] = hourly_counts['DateTime'].map({0: '12AM', 1: '1AM', 2: '2AM', 3: '3AM', 4: '4AM', 5: '5AM',
                                                               6: '6AM', 7: '7AM', 8: '8AM', 9: '9AM', 10: '10AM', 11: '11AM',
                                                               12: '12PM', 13: '1PM', 14: '2PM', 15: '3PM', 16: '4PM', 17: '5PM',
                                                               18: '6PM', 19: '7PM', 20: '8PM', 21: '9PM', 22: '10PM', 23: '11PM'})
# Plot the chart using plotly.express
    fig = px.line(hourly_counts, x='DateTime', y='Message Count', title='Messages by Hour')
    fig.show()


def pre_process(dataframe):
    pattern = r'[^a-zA-Z\s]|\s+[a-zA-Z]\s+|\s+'
# apply the regex pattern to the 'message' column, remove punctuations and emojis
    dataframe['New Message'] = dataframe['Message'].apply(lambda x: re.sub(pattern, ' ', x))

# convert the 'message' column to lowercase
    dataframe['New Message'] = dataframe['New Message'].str.lower()

    # Filter out the rows containing these texts
    dataframe = dataframe[~dataframe['Message'].str.contains('omitted|deleted')]
    filter_func = lambda s: ' '.join([w for w in s.split() if len(w) >= 6])
# apply the filter function to the specified column
    dataframe['New Message'] = dataframe['New Message'].apply(filter_func)
    dataframe['New Message'].replace('',np.nan,regex = True, inplace=True)
    dataframe.dropna(inplace=True)
    data_new = dataframe.loc[dataframe["New Message"].str.count(" ") >= 10]
    data_new.reset_index(inplace=True, drop=True)
    print('Preprocessing done')
    return data_new

def topic_modelling():
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'dey', 'go', 'na', 'people', 'even', 'make', 'know', 'one', 'still', 'like', 'say', 'time',
                       'sha', 'ooo', 'sure', 'first', 'person', 'want', 'wey', 'take', 'give', 'ni', 'day', 'problem', 
                       'Abeg',    'Oya',    'Nawa',    'Chop',    'Shishi',    'Wahala',    'Sabi',
                       'Jenifa',    'Soro',    'Kai',    'Banga',    'Wetin',    'Gbege',    'Kolo',    'Belle',    'Pikin',    'Gist',    'Ehen', 
                       'Shine',    'Oyinbo',    'Katakata',    'Hustle',    'Padi',    'Ikebe',    'Naija',    'Ojoro',    'Jollof',    'Jasi',   
                       'Waka',    'Ogbonge',    'Kpatakpata',    'Mumu',    'Orobo',    'Skelewu',    'Amebo',    'Aproko',    'Sisi',    'Yawa',  
                       'Chai',    'Abi',    'Gbese',    'Gbera',    'Go-slow',    'Gbe body e',    'Kpanlogo',    'Lai-lai',
                       'Palava',    'Suo',    'Totori',    'Zobo'])
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                 if word not in stop_words] for doc in texts]
    data = df['New Message'].tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    #print(data_words[:1][0][:30])
# Create Dictionary
    id2word = corpora.Dictionary(data_words)
# Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
# View
    num_topics = 10
# Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    doc_lda = lda_model[corpus]
    def get_topic_keywords(lda_model):
        topic_keywords = {}
        for topic_id in range(lda_model.num_topics):
            topic_keywords[topic_id] = [word for word, prob in lda_model.show_topic(topic_id)]
        print(topic_keywords)
        return topic_keywords
# Visualize the topics
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    return LDAvis_prepared
