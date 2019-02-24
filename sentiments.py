# -*- coding: utf-8 -*-
import pandas as pd
import re
import os
import requests
from textblob_de import TextBlobDE as TextBlob

# TODO reassign it all to big dataframe

# Load dataframe
# dataframe = pd.read_pickle(os.path.join('data', 'filtered_documents.pkl'))

# Save and pickle part of the file for easy testing
# dataframe.head(100).to_pickle("data/testing_dataframe.pkl")
dataframe = pd.read_pickle(os.path.join('data', 'testing_dataframe.pkl'))

# Further clean data from gibberish?


# Sentiment analysis
# Get sentiments with TextBlob

def get_sentiments_blob(sentence):
    text = TextBlob(sentence)
    return text.sentiment[0] + 1


def get_polarity_by_newspaper(corpus):
    sentiment_list = []
    for daily_paper in corpus.values:
        sentiment_score = 0
        sentence_list = re.split('[?.!]', daily_paper)
        for sentence in sentence_list:
            sentiment_score += get_sentiments_blob(sentence)
        sentiment_list.append(sentiment_score/len(sentiment_list))
    return sentiment_list


# TODO sometimes sentences are dates, titles, etc


dataframe["polarity_by_newspaper"] = get_polarity_by_newspaper(dataframe.text)
dataframe.to_pickle("data/dataframe_with_polarity.pkl")


