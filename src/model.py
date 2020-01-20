from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings
from typing import List
from flair.trainers import ModelTrainer


def get_corpus_and_tagger():
    columns = {0: 'text', 1: 'ner'}

    data_folder = 'data/'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train_IOB_Format_file.txt',
                                  dev_file="train_IOB_Format_file.txt")

    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embedding_types: List[TokenEmbeddings] = [

        #     WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        #     CharacterEmbeddings(),

        # comment in these lines to use flair embeddings.
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    return corpus, tagger


def train_model(corpus, tagger):
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('models/',
                  learning_rate=0.1,
                  mini_batch_size=16,
                  max_epochs=10)


if __name__ == '__main__':
    corpus, tagger = get_corpus_and_tagger()
    train_model(corpus, tagger)
