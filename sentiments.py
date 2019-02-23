# -*- coding: utf-8 -*-
import pandas as pd
import re
import os
import requests
from polyglot.text import Text


# TODO reassign it all to big dataframe

# Load dataframe
# dataframe = pd.read_pickle(os.path.join('data', 'filtered_documents.pkl'))

# Save and pickle part of the file for easy testing
# dataframe.head(100).to_pickle("data/testing_dataframe.pkl")
dataframe = pd.read_pickle(os.path.join('data', 'testing_dataframe.pkl'))

# # Tokenize data
# german_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
# dataframe.text = dataframe.text.apply(lambda text_in_dataframe: german_tokenizer.tokenize(text_in_dataframe))


# Further clean data from gibberish?


# Sentiment analysis

def get_sentences(corpus):
    text_list = []
    for daily_paper in corpus.values:
        sentence = re.split('[?.!]', " ".join(daily_paper.split()))
        text_list.extend(sentence)
    return text_list

# TODO sometimes sentences are dates, titles, etc

all_sentences = get_sentences(dataframe.text)
# }

# Get sentiments with ipublia servers
def get_sentiments(sentences):
    response = requests.post("http://127.0.0.1:5000/predict", json={
        "texts": sentences
    })
    return response.json()

# get_sentiments(all_sentences)


# Get sentiments with polyglot
all_sentences_tpl = tuple(all_sentences)

def get_sentiments_poly(sentences):
    text = Text(sentences)
    for word in text.words:
        print(word.polarity)

get_sentiments_poly(all_sentences[:10])