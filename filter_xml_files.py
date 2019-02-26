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
import spacy
from symspellpy.symspellpy import SymSpell, Verbosity


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


def clean_text(articles_df: pd.DataFrame):
    """
    Cleans extracted text - removes special characters etc.
    :param articles_df:
    :return:
    """

    step_size = 100

    pbar = tqdm(total=math.ceil(len(articles_df) / step_size))
    for i in range(0, len(articles_df), step_size):
        slice_df = articles_df[i:i + step_size]

        # &#xA: Newline character.
        slice_df.text = slice_df.text.str[11:-13]
        # Remove special characters.
        slice_df.text = slice_df.text.str.replace(r"«|»|&#xA|\^|&gt;|&lt;|;|&apos;|\*|■|&quot", " ")
        # Remove single characters.
        slice_df.text = slice_df.text.str.replace(r'(?i)\b[a-z]\b', " ")
        # Collapes multiple whitespaces to one.
        slice_df.text = slice_df.text.str.replace(r" +", " ")

        # Update text values.
        articles_df.iloc[i:i + step_size, articles_df.columns.get_loc("text")].text = slice_df.text.values

        pbar.update(1)

    return articles_df


def extract_zip_file():
    """
    Extract relevant files from .zip files.
    :return: 
    """
    zip_file = zipfile.ZipFile('/media/raphael/Elements/9200338.zip')

    articles_df = filter_files(zip_file)
    articles_df.to_pickle(path="/media/raphael/Elements/filtered_documents.pkl")


def preprocess_corpus(articles_df: pd.DataFrame) -> list:
    """
    Preprocess corpus before creation of DTM.
    :param articles_df:
    :return:
    """

    nlp = spacy.load("de")

    ######################################
    # Define and remove stop words,
    # tokenize text w/o stop words.
    ######################################

    nlp.Defaults.stop_words |= {"Nr"}

    print("replacing special chars and single-char words")
    articles_df.text = articles_df.text.str.replace(r"[0-9]+|\\|,|\.|:|!\?|-|►|—|&amp|Dr|\|„|/|!|\?|\+|\|„|\(|\)", " ")
    articles_df.text = articles_df.text.str.replace(r"\b.\b", " ")
    articles_df.text = articles_df.text.str.replace(r"\s+", " ")

    print("removing stop words")
    articles_df.text = articles_df.text.apply(
        lambda x: [word for word in x.split(" ") if word not in nlp.Defaults.stop_words]
    )

    ######################################
    # todo use word frequencies (ideally
    #  around 1900 -> n-grams data) and
    #  symspellpy to fix or remove
    #  unknown words.
    ######################################

    max_edit_distance_dictionary = 2
    prefix_length = 7
    symspell_instance_filename = "symspell_instance.pkl"

    #  Dictionary from https://github.com/hermitdave/FrequencyWords/blob/master/content/2016/de/de_full.txt.
    # if not os.path.isfile(symspell_instance_filename):
    #     print("create symspell dictionary")
    #     sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    #     assert sym_spell.create_dictionary(corpus="word_frequencies_de.txt"), "Corpus file not found"
    #
    #     with open(symspell_instance_filename, 'wb') as symspell_instance_file:
    #         pickle.dump(sym_spell, symspell_instance_file)
    # else:
    #     print("load symspell dictionary")
    #     with open(symspell_instance_filename, 'rb') as symspell_instance_file:
    #         sym_spell = pickle.load(symspell_instance_file)

    # Remove words for which no suitable replacements can be found (note that we don't correct any OCR mistakes here).
    print("removing unknown words")
    pbar = tqdm(total=len(articles_df))
    with open("word_frequencies_de.txt", "r") as word_file:
        word_set = {line.replace("\n", "").split()[0].lower().strip() for line in word_file}
    articles_df.text = articles_df.text.apply(lambda x: remove_gibberish(x, None, word_set, pbar))
    pbar.close()


def remove_gibberish(
        text: list, sym_spell: SymSpell, word_set: set, pbar: tqdm, max_edit_distance_lookup: int = 2
) -> list:
    """
    Removes words without alternative suggestions (indicating that they are not in the 50000 most common german
    words and hence actually gibberish. Typos should be covered by allowing a certain edit distance.
    :param text:
    :param sym_spell:
    :param word_set:
    :param pbar:
    :param max_edit_distance_lookup:
    :return:
    """

    # Using symspellpy takes ~40s per article, for our purposes too slow.
    # words = []
    # for word in text:
    #     res = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance_lookup)
    #     if len(res) and res[0].distance < max_edit_distance_lookup:
    #         words.append(word)

    pbar.update(1)

    # Keep only words that are in our set of known words.
    return [word for word in text if word.lower().strip() in word_set]


def create_dynamic_topic_model(articles_df: pd.DataFrame) -> list:
    """
    Creates DTM on text.
    :param articles_df:
    :return:
    """
    path_to_dtm_binary = "/home/raphael/Development/dtm/dtm/"

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(articles_df.text.values)
        exit()

    # remove stop words
    # tokenize by splitting on ' '
    # stem/lemmatize
    # create bigrams/trigrams

    # create gensim corpus and dictionary

    # todo run DTM

    return []


if __name__ == '__main__':
    logging.basicConfig(format='%)s - %(message)s', level=logging.INFO)

    file_location = "~/Development/data/dh-hackathon/filtered_documents.pkl"
    df = pd.read_pickle(path=file_location)
    
    # logging.info('Cleaning text.')
    # df = clean_text(df)
    # df.to_pickle(path=file_location)

    preprocess_corpus(df)
    df.to_pickle(path="~/Development/data/dh-hackathon/cleaned_filtered_documents.pkl")
