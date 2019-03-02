"""
Utilities for filtering XML files by date range.
"""
import logging
import math
import os
import pickle
import time
import zipfile

from tqdm import tqdm
import pandas as pd
import gensim
from gensim.test import utils as gensim_test_utils
import spacy
import numpy as np
import spacy.lang.de
from nltk.stem.cistem import Cistem


def filter_files(archive_file: zipfile.ZipFile) -> pd.DataFrame:
    filenames = archive_file.namelist()
    pbar = tqdm(total=len(filenames))
    documents = []

    for filename in archive_file.namelist():
        with archive_file.open(filename) as xml_file:
            try:
                # Extract date.
                info_url = str([next(xml_file) for x in range(4)][-1]).split("\"")[1].split("/")
                year = int(info_url[11])

                if 1913 <= year <= 1919:
                    while True:
                        if "<edm:FullTextResource" in str(next(xml_file))[1:]:
                            next(xml_file)
                            text = next(xml_file).decode(encoding='UTF-8', errors='strict')[1:]
                            documents.append({
                                "newspaper": info_url[10],
                                "year": year,
                                "month": int(info_url[12]),
                                "day": int(info_url[13]),
                                "filename": filename,
                                "text": text
                            })
                            break

            except Exception as e:
                print("Failed at file " + filename, e)

        pbar.update(1)

    return pd.DataFrame(documents)


def extract_zip_file():
    """
    Extract relevant files from .zip files.
    :return: 
    """
    zip_file = zipfile.ZipFile('/media/raphael/Elements/9200338.zip')

    articles_df = filter_files(zip_file)
    articles_df.to_pickle(path="/media/raphael/Elements/filtered_documents.pkl")


def clean(articles_df: pd.DataFrame, word_occurence_threshold: int = 5):
    """
    Preprocess corpus before creation of DTM.
    :param articles_df:
    :param word_occurence_threshold:
    :return:
    """

    nlp = spacy.load("de", disable=['parser', 'ner'])
    nlp.Defaults.stop_words |= {
        "Nr", "herr", "frau", "abend", "mann", "sucht", "nickt", "januar", "jänner", "februar", "märz", "april", "mai",
        "juni", "juli", "august", "september", "oktober", "november", "dezember", "montag", "dienstag", "mittwoch",
        "donnerstag", "freitag", "samstag", "sonntag", "letzten", "bill", "hamburg", "hamburger", "deutsch",
        "deutscher", "deutsche", "deutschland", "deutschen", "berlin", "bereit", "lassen", "ihren", "find", "meldet",
        "gestern", "karl", "paul", "otto", "hermann", "schwer", "sept", "herren", "friedrich", "wilhelm", "heinrich",
        "frage", "sofort", "voll", "herrn"
    }
    interval_size = 250
    num_iter = math.ceil(len(articles_df) / interval_size)

    ######################################
    # Define and remove stop words,
    # tokenize text w/o stop words.
    ######################################

    print("replacing special chars and single-char words")
    pbar = tqdm(total=num_iter)
    for i in range(0, len(articles_df), interval_size):
        articles_df[i:i + interval_size].text = articles_df[i:i + interval_size].text.str.replace(
            r"[^a-zA-ZäüöÄÜÖß]+", ' '
        )
        articles_df[i:i + interval_size].text = articles_df[i:i + interval_size].text.str.replace(r"\b.\b", " ")
        articles_df[i:i + interval_size].text = articles_df[i:i + interval_size].text.str.replace(r"\s+", " ")
        pbar.update(1)
    pbar.close()

    ######################################
    # todo use word frequencies (ideally
    #  around 1900 -> n-grams data) and
    #  symspellpy to fix or remove
    #  unknown words.1
    ######################################

    # Remove words for which no suitable replacements can be found (note that we don't correct any OCR mistakes here).
    print("removing stop words and unknown words")
    with open("word_frequencies_de.txt", "r") as word_file:
        word_set = {
            line.replace("\n", "").split()[0].lower().strip()
            for line in word_file
            # Only keep words with at least word_occurence_threshold appearances in dictionary.
            if int(line.replace("\n", "").split()[1]) >= word_occurence_threshold
        }

        pbar = tqdm(total=num_iter)
        for i in range(0, len(articles_df), interval_size):
            articles_df[i:i + interval_size].text = articles_df[i:i + interval_size].text.apply(
                lambda x: " ".join([
                    word for word in x.split()
                    if word.lower().strip() in word_set and
                    word.lower().strip() not in nlp.Defaults.stop_words and
                    len(word) >= 4
                ])
            )
            pbar.update(1)
        pbar.close()


def stem(articles_df: pd.DataFrame):
    """
    Stems list of articles with list of words.
    param texts:
    """
    stemmer = Cistem()

    def stem_article(stemmer: Cistem, text: str, pbar: tqdm) -> str:
        pbar.update(1)
        return " ".join(stemmer.stem(word) for word in text.split())

    progress_bar = tqdm(total=len(articles_df))
    articles_df.text = articles_df.text.apply(lambda x: stem_article(stemmer, x, progress_bar))
    progress_bar.close()


def concatenate_bigrams(articles: list, force_recompilation: bool):
    """
    Concatenates most common 2-bigrams with _.
    :param articles:
    :param force_recompilation:
    """
    interval_size = 250
    num_iter = math.ceil(len(articles) / interval_size)

    bigrams_phraser_filepath = "bigrams_phraser.pkl"
    if not os.path.isfile(bigrams_phraser_filepath) or force_recompilation:
        bigram = gensim.models.phrases.Phraser(
            gensim.models.phrases.Phrases(articles, min_count=5, threshold=100, progress_per=100)
        )
        with open(bigrams_phraser_filepath, "wb") as phraser_file:
            pickle.dump(bigram, phraser_file)
    else:
        with open(bigrams_phraser_filepath, "rb") as phraser_file:
            bigram = pickle.load(phraser_file)

    pbar = tqdm(total=num_iter)
    for i in range(0, len(articles), interval_size):
        articles[i:i + interval_size] = [bigram[article] for article in articles[i:i + interval_size]]
        pbar.update(1)
    pbar.close()


def create_dictionary(articles: list, force_recompilation: bool) -> gensim.corpora.Dictionary:
    """
    Creates gensim dictionary from list of tokenized article texts.
    :param articles:
    :param force_recompilation:
    :return:
    """

    dict_filepath = "id2word.dict"
    if not os.path.isfile(dict_filepath) or force_recompilation:
        id2word = gensim.corpora.Dictionary(articles)
        id2word.save(dict_filepath)
    else:
        id2word = gensim.corpora.Dictionary.load(fname=dict_filepath)

    return id2word


def transform_texts_to_corpus_ids(
        articles: list, id2word: gensim.corpora.Dictionary, force_recompilation: bool
) -> list:
    """
    Replaces words in articles with correspondig token IDs.
    :param articles:
    :param id2word:
    :param force_recompilation:
    :return:
    """

    interval_size = 250
    num_iter = math.ceil(len(articles) / interval_size)

    corpus_filepath = "corpus.mm"
    if not os.path.isfile(corpus_filepath) or force_recompilation:
        pbar = tqdm(total=num_iter)
        for i in range(0, len(articles), interval_size):
            articles[i:i + interval_size] = [
                id2word.doc2bow(article) for article in articles[i:i + interval_size]
            ]
            pbar.update(1)
        pbar.close()
        gensim.corpora.MmCorpus.serialize(corpus_filepath, articles)
    else:
        articles = gensim.corpora.MmCorpus(gensim_test_utils.datapath(corpus_filepath))

    return articles


def preprocess_corpus(filepath: str, force_recompilation: bool = False):
    """
    Creates DTM on text.
    :param filepath:
    :param force_recompilation:
    :return:
    """

    articles = pd.read_pickle(path=filepath).text.values.tolist()
    interval_size = 250
    num_iter = math.ceil(len(articles) / interval_size)

    # Tokenize words.
    print("tokenizing words")
    pbar = tqdm(total=num_iter)
    for i in range(0, len(articles), interval_size):
        articles[i:i + interval_size] = [
            [word.replace(")", "") for word in article.split()]
            for article in articles[i:i + interval_size]
        ]
        pbar.update(1)
    pbar.close()

    # Mark bigrams.
    print("marking bigrams")
    concatenate_bigrams(articles, force_recompilation)
    
    # Creating gensim dictionary.
    print("creating dictionary")
    id2word = create_dictionary(articles, force_recompilation)

    # Creating gensim corpus.
    print("creating corpus")
    transform_texts_to_corpus_ids(articles, id2word, force_recompilation)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger("preprocessing")

    #file_location = "~/Development/data/dh-hackathon/filtered_documents.pkl"
    #df = pd.read_pickle(file_location).sort_values(by=["year", "month", "day"])
    #clean(df, word_occurence_threshold=15)
    #df.to_pickle("~/Development/data/dh-hackathon/cleaned_filtered_documents-tau15.pkl")
    # stem(df)
    preprocess_corpus("~/Development/data/dh-hackathon/cleaned_filtered_documents-tau15.pkl")

