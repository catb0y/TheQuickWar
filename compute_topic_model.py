from pprint import pprint
import pandas as pd
import gensim
from gensim.test import utils as gensim_test_utils


def create_dynamic_topic_model(corpus_filepath: str, dict_filepath: str, time_slices: list):
    """
    Creates topic models based on preprocessed data.
    :param corpus_filepath:
    :param dict_filepath:
    :param time_slices:
    :return:
    """

    path_to_dtm_binary = "/home/raphael/Development/dtm/dtm/"

    corpus = gensim.corpora.MmCorpus(gensim_test_utils.datapath(corpus_filepath))
    id2word = gensim.corpora.Dictionary.load(fname=dict_filepath)

    print("generating model")
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=10,
        random_state=100,
        update_every=1,
        chunksize=1000,
        passes=20,
        alpha='auto',
        per_word_topics=True
    )

    pprint(lda_model.print_topics())


if __name__ == '__main__':
    create_dynamic_topic_model(
        corpus_filepath="/home/raphael/Development/dh-hackathon/corpus.mm",
        dict_filepath="/home/raphael/Development/dh-hackathon/id2word.dict",
        # Get time_slices sorted by year and month, ascendingly.
        # Important: Has to match with sequence in which documents where fed to corpus generation procedure (i. e. sort
        # dataframe the same way before generating corpus)."
        time_slices=pd.read_pickle(
            "~/Development/data/dh-hackathon/cleaned_filtered_documents-tau15.pkl"
        ).groupby(["year", "month"]).size().values.tolist()
    )