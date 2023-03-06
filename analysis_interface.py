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
import pickle 
import pyLDAvis
import os
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import re
import string
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim import corpora, models
#@st.experimental_memo
#peterobi = pd.read_csv('peterobi.csv')
#atiku = pd.read_csv('atiku.csv')
#bat = pd.read_csv('BAT.csv')
import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
#import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import zipfile
import io

st.set_page_config(
    page_title="Chatistics",
    page_icon="📊",
    layout="wide",
)

# dashboard title
st.title('Chatistics 📈')
st.markdown("Deep Exploratory Chat Analysis")

@st.cache
def import_chat(file):
    name = os.path.splitext(file.name)
    name = name[1].replace(".", "")
    if name == "zip":
        with zipfile.ZipFile(file) as zip_file:
            # Assume there is only one file in the zip archive
            file_name = zip_file.namelist()[0]
            with zip_file.open(file_name) as chat_file:
                # Read the file
                lines = io.TextIOWrapper(chat_file, encoding='utf-8').read().splitlines()
        # Read the file
    elif name == "txt":
        lines = file.read().decode('utf-8').splitlines()
    
    # Extract the data using regular expressions
    data = []
    for line in lines:
        # Match both formats of datetime strings
        match = re.findall(r'\[(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)\]\s+(.*?):\s+(.*)', line.strip())
        if match:
            data.append(match[0])
    df = pd.DataFrame(data, columns=['DateTime', 'Author', 'Message'])
    df.drop([0, 1, 2], axis='index', inplace=True)
    df = df[~df['Message'].str.contains('omitted|deleted|changed group')]
    df['Message'].replace('',np.nan,regex = True, inplace=True)
    df.dropna(inplace=True)
    # Create a new dataframe with only valid rows
    valid_rows = []
    for i, row in df.iterrows():
        try:
            date_time = pd.to_datetime(row['DateTime'], format='%m/%d/%y, %I:%M:%S %p')
        except ValueError:
            date_time = pd.to_datetime(row['DateTime'], format='%d/%m/%Y, %H:%M:%S')
        row['Date'] = date_time.date()
        row['Time'] = date_time.time()
        valid_rows.append(row)
    df = pd.DataFrame(valid_rows)
    df['Message Character Count'] = df['Message'].apply(lambda x: len(x))
    df.reset_index(inplace=True, drop=True)
    return df

def plot_message(data):
    #data = import_chat(file)
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
    return st.plotly_chart(fig, use_container_width=True)

@st.cache
def download_data():
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

def character_count(data):
    #data = import_chat(file)
# Calculate the total character count for each author
    author_character_counts = data.groupby('Author')['Message Character Count'].sum()
# Sort the authors by their total character count in descending order
    sorted_authors_1 = author_character_counts.sort_values(ascending=False)
    sorted_authors = sorted_authors_1.iloc[:10]
# Create a dataframe with the author and message count data
    data_character = pd.DataFrame({'Author': sorted_authors.index, 'Character Count': sorted_authors.values})
# Create the plot using plotly.express
    fig = px.bar(data_character, x='Character Count', y='Author', orientation='h',
                 labels={'Character Count': 'Character Count', 'Author': 'Author'},
                 title='Top 10 Authors by Character Count')
    return sorted_authors_1

@st.cache
def count_chat_volume(data):
    message_counts = data.groupby('Date').size()
    # Sort the message counts in descending order
    sorted_counts = message_counts.sort_values(ascending=False)
# Get the two dates with the highest message counts
    top_dates = sorted_counts.index[:2]
# Print the top dates
    return top_dates

def chat_volume_over_time(data):
    message_counts = data.groupby('Date').size()
    # Sort the message counts in descending order
    sorted_counts = message_counts.sort_values(ascending=False)
# Get the two dates with the highest message counts
    top_dates = sorted_counts.index[:2]
# Print the top dates
    print(f"The two dates with the highest messages are {top_dates[0]} and {top_dates[1]}.")

# Plot the message counts over time
# create a line chart with plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=message_counts.index, y=message_counts.values, mode='lines'))
# add titles and axis labels
    fig.update_layout(title='Chat Volume Over Time',
                      xaxis_title='Date',
                      yaxis_title='Number of Messages')
# rotate x-axis labels
    fig.update_layout(xaxis_tickangle=-45)
    return st.plotly_chart(fig, use_container_width=True)

# show the plot

def pie_chart(data):
# Calculate the total character count for each author
    #data = character_count(data)
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
    st.plotly_chart(fig, use_container_width=True)

def line_by_hour(data):
    #data = character_count(data)
    try:
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%y, %I:%M:%S %p')
    except ValueError:
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y, %H:%M:%S')
# Group the messages by hour
    hourly_counts = data.groupby(data['DateTime'].dt.hour)['Message'].count().reset_index(name='Message Count')
    hourly_counts['DateTime'] = hourly_counts['DateTime'].map({0: '12AM', 1: '1AM', 2: '2AM', 3: '3AM', 4: '4AM', 5: '5AM',
                                                               6: '6AM', 7: '7AM', 8: '8AM', 9: '9AM', 10: '10AM', 11: '11AM',
                                                               12: '12PM', 13: '1PM', 14: '2PM', 15: '3PM', 16: '4PM', 17: '5PM',
                                                               18: '6PM', 19: '7PM', 20: '8PM', 21: '9PM', 22: '10PM', 23: '11PM'})
# Plot the chart using plotly.express
    fig = px.line(hourly_counts, x='DateTime', y='Message Count', title='Messages by Hour')
    return st.plotly_chart(fig, use_container_width=True)

def collocation_extraction(data):
    df = data
    df['Message'] = df['Message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    corpus = ' '.join(df['Message'])
# Tokenize the corpus
    tokens = nltk.word_tokenize(corpus)
# Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
# Create a bigram collocation finder
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
# Filter out bigrams that occur less than 3 times
    finder.apply_freq_filter(3)
# Compute the top 10 bigrams using the chi-squared measure
    top_bigrams = finder.nbest(bigram_measures.chi_sq, 10)

# Create a dictionary of collocations and their frequencies
    collocations_freq_dict = {}
    for bigram in top_bigrams:
        collocations_freq_dict[' '.join(bigram)] = tokens.count(bigram[0]) + tokens.count(bigram[1])

# Create a Plotly bar chart with x and y values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(collocations_freq_dict.keys()), y=list(collocations_freq_dict.values())))

# Add title and axis labels
    fig.update_layout(title=f"Top Collocations in the conversation",
                      xaxis_title="Collocations",
                      yaxis_title="Frequency")

    return st.plotly_chart(fig, use_container_width=True)

def percentage_author_contribution(data, selected_author):
# Calculate the total number of messages
    total_messages = len(data)
# Calculate the number of messages sent by the selected author
    selected_author_messages = len(data[data['Author'] == selected_author])
# Calculate the percentage of messages sent by the selected author
    selected_author_percentage = selected_author_messages / total_messages * 100
# Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=[selected_author, 'Total Conversation'],
                                 values=[selected_author_percentage, 100 - selected_author_percentage],
                                 hole=0.5)])
    
    return st.plotly_chart(fig, use_container_width=True)

def emotions_analysis_group(data, selected_author):
    df = data
    # Load pre-trained emotion classification model
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Define emotion labels
    emotion_labels = ['anger', 'joy', 'optimism', 'sadness']
    # Define the selected author
    # Filter messages by selected author
    author_messages = df['Message'].tolist()
# Predict the emotions for each message by the selected author
    emotions = []
    with st.spinner('Performing deep emotion analysis...'):
        for message in author_messages:
            inputs = tokenizer(message, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            predicted_emotion = emotion_labels[int(torch.argmax(outputs.logits))]
            emotions.append(predicted_emotion)
    st.success('Completed')
# Compute the frequency of each emotion
    freq_dict = {}
    for emotion in emotions:
        freq_dict[emotion] = freq_dict.get(emotion, 0) + 1
    # Create a Pie chart with the emotions frequency
    fig = go.Figure(data=[go.Pie(labels=list(freq_dict.keys()), values=list(freq_dict.values()), hole=0.5)])

    # Add title and subtitle
    fig.update_layout(title={'text': "Emotions of " + selected_author + " in WhatsApp Chat", 'y':0.9})
    return st.plotly_chart(fig, use_container_width=True)

def sentiment_analysis_group(data, selected_author):
# Load the dataset
    df = data
# Select the author for sentiment analysis
# Filter the messages of the selected author
    author_messages = df['Message'].tolist()
# Initialize the Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
# Compute the sentiment scores for each message
    sentiment_scores = [analyzer.polarity_scores(message) for message in author_messages]
# Compute the overall sentiment for the selected author
    overall_sentiment = sum([score['compound'] for score in sentiment_scores]) / len(sentiment_scores)
# Compute the percentage of positive, negative, and neutral messages
    num_messages = len(sentiment_scores)
    num_positive = len([score for score in sentiment_scores if score['compound'] > 0])
    num_negative = len([score for score in sentiment_scores if score['compound'] < 0])
    num_neutral = num_messages - num_positive - num_negative
    positive_percentage = (num_positive / num_messages) * 100
    negative_percentage = (num_negative / num_messages) * 100
    neutral_percentage = (num_neutral / num_messages) * 100

# Create a Plotly pie chart
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_percentage, negative_percentage, neutral_percentage]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title={'text': f'Sentiment Analysis for {selected_author}', 'y':0.9})
    # Show the plot
    return st.plotly_chart(fig, use_container_width=True)

######################################Linguistic pattern extraction

def word_frequency(data, selected_author):
    author_messages = data.loc[data['Author'] == selected_author, 'Message'].tolist()

# Clean and preprocess the text data
    custom_stop_words = ['omitted audio', 'omitted']
    stop_words = set(stopwords.words('english') + custom_stop_words)


    preprocessed_messages = []
    for message in author_messages:
        message = message.lower()
        words = nltk.word_tokenize(message)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        preprocessed_messages.append(words)

# Extract the frequency of top 10 words
    word_counts = Counter([word for message in preprocessed_messages for word in message])
    top_10_words = dict(word_counts.most_common(10))

# Visualize the results using a bar chart
    fig = go.Figure(go.Bar(x=list(top_10_words.keys()), y=list(top_10_words.values())))
    fig.update_layout(title=f"Top 10 Words Used by {add_selectbox}")
    return st.plotly_chart(fig, use_container_width=True)

def collocation_extraction_by_author(data, selected_author):
    df = data
    df['Message'] = df['Message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    corpus = ' '.join(df.loc[df['Author'] == selected_author]['Message'])
# Tokenize the corpus
    tokens = nltk.word_tokenize(corpus)
# Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
# Create a bigram collocation finder
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
# Filter out bigrams that occur less than 3 times
    finder.apply_freq_filter(3)
# Compute the top 10 bigrams using the chi-squared measure
    top_bigrams = finder.nbest(bigram_measures.chi_sq, 10)
# Create a dictionary of collocations and their frequencies
    collocations_freq_dict = {}
    for bigram in top_bigrams:
        collocations_freq_dict[' '.join(bigram)] = tokens.count(bigram[0]) + tokens.count(bigram[1])
# Create a Plotly bar chart with x and y values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(collocations_freq_dict.keys()), y=list(collocations_freq_dict.values())))

# Add title and axis labels
    fig.update_layout(title=f"Top Collocations in WhatsApp Chat by {selected_author}",
                      xaxis_title="Collocations",
                      yaxis_title="Frequency")
    return st.plotly_chart(fig, use_container_width=True)

def sentiment_analysis(data, selected_author):
# Load the dataset
    df = data
# Select the author for sentiment analysis
# Filter the messages of the selected author
    author_messages = df[df['Author'] == selected_author]['Message'].tolist()
# Initialize the Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
# Compute the sentiment scores for each message
    sentiment_scores = [analyzer.polarity_scores(message) for message in author_messages]
# Compute the overall sentiment for the selected author
    overall_sentiment = sum([score['compound'] for score in sentiment_scores]) / len(sentiment_scores)
# Compute the percentage of positive, negative, and neutral messages
    num_messages = len(sentiment_scores)
    num_positive = len([score for score in sentiment_scores if score['compound'] > 0])
    num_negative = len([score for score in sentiment_scores if score['compound'] < 0])
    num_neutral = num_messages - num_positive - num_negative
    positive_percentage = (num_positive / num_messages) * 100
    negative_percentage = (num_negative / num_messages) * 100
    neutral_percentage = (num_neutral / num_messages) * 100

# Create a Plotly pie chart
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_percentage, negative_percentage, neutral_percentage]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title={'text': f'Sentiment Analysis for {selected_author}', 'y':0.9})
    # Show the plot
    return st.plotly_chart(fig, use_container_width=True)

def emotions_analysis(data, selected_author):
    df = data
    # Load pre-trained emotion classification model
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Define emotion labels
    emotion_labels = ['anger', 'joy', 'optimism', 'sadness']
    # Define the selected author
    # Filter messages by selected author
    author_messages = df[df['Author'] == selected_author]['Message'].tolist()
# Predict the emotions for each message by the selected author
    emotions = []
    with st.spinner('Performing deep emotion analysis...'):
        for message in author_messages:
            inputs = tokenizer(message, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            predicted_emotion = emotion_labels[int(torch.argmax(outputs.logits))]
            emotions.append(predicted_emotion)
    st.success('Completed')
# Compute the frequency of each emotion
    freq_dict = {}
    for emotion in emotions:
        freq_dict[emotion] = freq_dict.get(emotion, 0) + 1
    # Create a Pie chart with the emotions frequency
    fig = go.Figure(data=[go.Pie(labels=list(freq_dict.keys()), values=list(freq_dict.values()), hole=0.5)])

    # Add title and subtitle
    fig.update_layout(title={'text': "Emotions of " + selected_author + " in WhatsApp Chat", 'y':0.9})
    return st.plotly_chart(fig, use_container_width=True)
########################################Function declarations finished#########################################
uploaded_file = st.file_uploader("Upload your exported WhatsApp Zip Chat File", key='files')
#click = st.button('Analyse', key="click1")

# creating a single-element container
# creating a single-element container
if uploaded_file is not None:
    data = import_chat(uploaded_file).copy()
    radio_list = data['Author'].unique()
    new_element = 'General'
    radio_list = np.concatenate(( [new_element], radio_list))
    
    #radio_list = data['Author'].unique()
    # Get the selected radio button from session state if available
    selected_radio = st.session_state.get('selected_radio', radio_list[0])
    add_selectbox = st.sidebar.radio("Select a chat member to analyze", radio_list, key='radio', index=radio_list.tolist().index(selected_radio))

    # Save the selected radio button to session state
    #st.session_state.selected_radio = add_selectbox

    # Display selected value
    #st.write(add_selectbox)
    if add_selectbox == 'General':
        st.write(f"<p style='font-size:20px'>Woah 😲, at a glance we have <b style='font-size:25px'>{data['Message'].count()}</b> messages, summing up to precisely \
            <b style='font-size:25px'>{data['Message Character Count'].sum()}</b> characters 🙀, here are the top folks with the most messages</p>", unsafe_allow_html=True)
        plot_message(data)
        #character_count(data)
        st.write(f"<p style='font-size:20px'>While we are here, here are the top conversationists at a glance based on character counts 📊. <blockquote>Let me break it down for you. These are the total number of keystrokes made on the phone's keyboard for each person</blockquote></p>", unsafe_allow_html=True)
        st.table(character_count(data).head(10))
        st.write(f"<p style='font-size:20px'>Curious to know the percentage contribution of each to the conversation? Voila!!</p>", unsafe_allow_html=True)
        pie_chart(data)
        st.write(f"<p style='font-size:20px'>Ya'll had some very serious conversations on <b>{count_chat_volume(data)[0]}</b> and <b>{count_chat_volume(data)[1]}</b>, chat volume was at it's highest</p>", unsafe_allow_html=True)
        chat_volume_over_time(data)
        st.write(f"<p style='font-size:20px'>And finally, we have conversation patterns presented as hour of the day members are most active</p>", unsafe_allow_html=True)
        line_by_hour(data)

        advanced_analysis = st.button('Advanced Analysis')
        if advanced_analysis:
            st.header('Collocation Extraction')
            st.write(f"<p style='font-size:15px'>Collocation is a linguistic term used to describe the co-occurrence of two or more words in a text that tend to appear together more often than would be expected by chance. Collocation extraction involves identifying such word combinations in a corpus of text.</p>", unsafe_allow_html=True)
            collocation_extraction(data)
            st.header('Sentiment Analysis')
            st.write(f"<p style='font-size:15px'>Sentiment analysis for Whatsapp Chats can reveal the emotional undertones of conversations and provide valuable insights into the feelings and attitudes of participants, enabling a better understanding of the dynamics of interpersonal relationships.</p>", unsafe_allow_html=True)
            sentiment_analysis_group(data, add_selectbox)
            st.write(f"<p style='font-size:18px'>This analysis is based on {data['Message'].count()} messages .</p>", unsafe_allow_html=True)
                      
            
    else:
        data1 = data.loc[data['Author'] == add_selectbox]    
        if data1['Message Character Count'].sum() > 1000:
            st.write(f"<p style='font-size:20px'>You did a number here, <b style='font-size:25px'>{data1['Message'].count()}</b> messages, summing up to precisely \
                <b style='font-size:25px'>{data1['Message Character Count'].sum()}</b> characters 🙀, Here is your total percentage contribution to the conversation</p>", unsafe_allow_html=True)
        else:
            st.write(f"<p style='font-size:20px'>What do we have here? <b style='font-size:25px'>{data1['Message'].count()}</b> messages, summing up to precisely \
                <b style='font-size:25px'>{data1['Message Character Count'].sum()}</b> characters, Here is your total percentage contribution to the conversation</p>", unsafe_allow_html=True)
        percentage_author_contribution(data, add_selectbox)
        if data1['Message'].count() < 10:
            st.write(f"<p style='font-size:20px'> It would seem our journey ends here 😔, you don't have enough contribution to the conversation for advanced pattern and linguistic analysis</p>", unsafe_allow_html=True)
        else:
            st.write(f"<p style='font-size:20px'>You contributed the most to the conversation on <b>{count_chat_volume(data1)[0]}</b></p>", unsafe_allow_html=True)
            chat_volume_over_time(data1)
            st.write(f"<p style='font-size:20px'>Here are your conversation patterns presented as hour of the day. These patterns predict, the time of the day you are most active in the conversation 🤳</p>", unsafe_allow_html=True)
            line_by_hour(data1)
            word_frequency(data, add_selectbox)
            advanced_analysis = st.button('Advanced Analysis')
            if advanced_analysis and data1['Message'].count() < 100:
                st.write(f"<p style='font-size:20px'> It would seem our journey ends here 😔, you don't have enough contribution to the conversation for advanced pattern and linguistic analysis</p>", unsafe_allow_html=True)
            elif advanced_analysis and data1['Message'].count() >= 100:
                st.header('Collocation Extraction')
                st.write(f"<p style='font-size:15px'>Collocation is a linguistic term used to describe the co-occurrence of two or more words in a text that tend to appear together more often than would be expected by chance. Collocation extraction involves identifying such word combinations in a corpus of text.</p>", unsafe_allow_html=True)
                collocation_extraction_by_author(data, add_selectbox)        
                st.header('Sentiment Analysis')
                st.write(f"<p style='font-size:15px'>Sentiment analysis for Whatsapp Chats can reveal the emotional undertones of conversations and provide valuable insights into the feelings and attitudes of participants, enabling a better understanding of the dynamics of interpersonal relationships.</p>", unsafe_allow_html=True)
                sentiment_analysis(data, add_selectbox)
                download_data()
                emotions_analysis(data, add_selectbox)
                st.write(f"<p style='font-size:18px'>This analysis is based on {data1['Message'].count()} messages .</p>", unsafe_allow_html=True)
        #st.write(f"<p style='font-size:20px'>😔 I am still working on personalized analysis, please check back </p>", unsafe_allow_html=True)
