import logging
import os
from pprint import pprint
import pandas as pd
import gensim
from gensim.test import utils as gensim_test_utils


def postprocess_topic_model(corpus_filepath: str, model_filepath: str, articles_filepath: str, dict_filepath: str):
    """
    Postprocesses results of finished topic model in preparation of visualization of topic relevance over time.
    :param corpus_filepath:
    :param model_filepath:
    :param articles_filepath:
    :param dict_filepath:
    :return:
    """
    corpus = gensim.corpora.MmCorpus(gensim_test_utils.datapath(corpus_filepath))
    model = gensim.models.LdaSeqModel.load(fname=model_filepath)
    id2word = gensim.corpora.Dictionary.load(fname=dict_filepath)
    df = pd.read_pickle(articles_filepath)

    # Store topics-in-article probabilities in dataframe.
    # df["topic_probs"] = [model.doc_topics(i) for i in range(0, len(corpus))]

    # df["topic_probs"] = [model.get_document_topics(doc) for doc in corpus]
    # Split topic probabilities into columns. Note: Assumes 6 topics.
    # df[["topic_" + str(i) for i in range(0, 6)]] = pd.DataFrame(df.topic_probs.values.tolist(), index=df.index)
    df = pd.read_pickle("tmp.pkl")
    def test(rec):
        print(rec)
        print(type(rec))
        exit()
    df[["topic_" + str(i) for i in range(0, 6)]] = df[["topic_" + str(i) for i in range(0, 6)]].apply(
        lambda x: test(x)
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df.head(10)[["topic_" + str(i) for i in range(0, 6)]])
    exit()
    # Group articles by year and month; compute average topic relevance over these attributes.
    res = df.drop("day", axis=1).groupby(["year", "month"]).mean()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df.head(10))


def create_dynamic_topic_model(corpus_filepath: str, dict_filepath: str, time_slices: list):
    """
    Creates topic models based on preprocessed data.
    :param corpus_filepath:
    :param dict_filepath:
    :param time_slices:
    :return:
    """

    corpus = gensim.corpora.MmCorpus(gensim_test_utils.datapath(corpus_filepath))
    id2word = gensim.corpora.Dictionary.load(fname=dict_filepath)

    # Create initial topic model.
    initial_lda_model_filepath = "model-5.lda"
    if not os.path.isfile(initial_lda_model_filepath):
        print("generating initial lda model")
        initial_lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=6,
            random_state=100,
            update_every=1,
            chunksize=1000,
            passes=20,
            alpha='auto',
            per_word_topics=True
        )

        pprint(initial_lda_model.print_topics())
        initial_lda_model.save(gensim_test_utils.datapath(initial_lda_model_filepath))
    else:
        print("loading initial lda model")
        initial_lda_model = gensim.models.LdaModel.load(fname=initial_lda_model_filepath)

    """
    1,
  '0.003*"Mark" + 0.002*"Regierung" + 0.002*"Stadt" + 0.001*"Kaiser" + '
  '0.001*"Verein" + 0.001*"Antrag" + 0.001*"Gesellschaft" + 0.001*"Leben" + '
  '0.001*"Sonnabend" + 0.001*"Meter"'),
 (2,
  '0.004*"Krieg" + 0.004*"Truppen" + 0.003*"England" + 0.003*"Front" + '
  '0.003*"Feind" + 0.003*"englischen" + 0.002*"englische" + 0.002*"Kriege" + '
  '0.002*"russischen" + 0.002*"Regierung"'),
 (3,
  '0.006*"London" + 0.005*"Dampfer" + 0.004*"Rotterdam" + 0.004*"Linie" + '
  '0.004*"York" + 0.003*"Aktien" + 0.003*"Bremen" + 0.003*"Antwerpen" + '
  '0.003*"Bank" + 0.003*"Makler"'),
 (4,
  '0.015*"gesucht" + 0.012*"Altona" + 0.006*"Gesucht" + 0.006*"billig" + '
  '0.006*"Mädchen" + 0.005*"verkaufen" + 0.003*"Wohn" + 0.003*"Part" + '
  '0.003*"verkauf" + 0.003*"Zimmer"'),
 (5,
  '0.004*"Dampfer" + 0.003*"Mill" + 0.003*"London" + 0.003*"Gesellschaft" + '
  '0.003*"Aktien" + 0.002*"Preise" + 0.002*"Dividende" + 0.002*"Krieg" + '
  '0.002*"Schiff" + 0.002*"York"')]
    """
    # Create dynamic topic model.
    print("creating dynamic topic model")
    dynamic_topic_model = gensim.models.LdaSeqModel(
        corpus=corpus,
        time_slice=time_slices,
        id2word=id2word,
        num_topics=6,
        initialize='gensim',
        # lda_model=initial_lda_model,
        passes=20,
        chunksize=1000
    )
    dynamic_topic_model.save("dynamic-topic-model.lda")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger("preprocessing")

    # create_dynamic_topic_model(
    #     corpus_filepath="/home/raphael/Development/dh-hackathon/corpus.mm",
    #     dict_filepath="/home/raphael/Development/dh-hackathon/id2word.dict",
    #     # Get time_slices sorted by year and month, ascendingly.
    #     # Important: Has to match with sequence in which documents where fed to corpus generation procedure (i. e. sort
    #     # dataframe the same way before generating corpus).
    #     time_slices=pd.read_pickle(
    #         "~/Development/data/dh-hackathon/cleaned_filtered_documents-tau15.pkl"
    #     ).groupby(["year", "month"]).size().values.tolist()
    # )

    postprocess_topic_model(
        corpus_filepath="/home/raphael/Development/dh-hackathon/corpus.mm",
        model_filepath="/home/raphael/Development/dh-hackathon/model-5.lda",
        dict_filepath="/home/raphael/Development/dh-hackathon/id2word.dict",
        articles_filepath="~/Development/data/dh-hackathon/cleaned_filtered_documents-tau15.pkl"
    )
