# -*- coding: utf-8 -*-
import os
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

# Load dataframe
dataframe = pd.read_pickle(os.path.join('data', 'cleaned_filtered_documents-tau15.pkl'))


# Sentiment analysis
# Use germanlex as a dictionary

def csv_to_polarity_dict():
    final_german_dictionary = {}
    with open('germanlex.csv', mode='r') as german_dictionary:
        for row in german_dictionary:
            row = row.split(',')
            if row[1] in ['POS', 'NEG', 'NEU']:
                final_german_dictionary[row[0]] = float(row[2]) if row[2] != "NA" else 0
        return final_german_dictionary


def get_polarity_by_newspaper(corpus):
    sentiment_list = []
    csv_to_polarity = csv_to_polarity_dict()
    for daily_paper in enumerate(corpus.values):
        sentiment_score = 0
        sentiment_count = 0
        word_division = daily_paper.split(' ')
        for word in word_division:
            if word in csv_to_polarity:
                sentiment_score += csv_to_polarity[word]
                sentiment_count += 1
        sentiment_list.append(sentiment_score/sentiment_count if sentiment_count != 0 else 0)
    return sentiment_list


# Create the dataframe with polarity
dataframe["polarity_by_newspaper"] = get_polarity_by_newspaper(dataframe.text)
dataframe.to_pickle("data/dataframe_with_polarity.pkl")


# Visualization
dataframe_with_polarity = pd.read_pickle('data/dataframe_with_polarity.pkl')
result = dataframe_with_polarity.groupby(["year", "month"]).mean()

trace1 = go.Scatter(
    x=[i for i in range(0, len(result.polarity_by_newspaper.values))],
    y=result.polarity_by_newspaper.values,
    fill='tozeroy',
    mode='none'
)

data = [trace1]
py.plot(data, filename='basic-area-no-bound')