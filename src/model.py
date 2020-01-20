from flair.data import Corpus
from flair.datasets import ColumnCorpus

def doWork():
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = ''

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='file_train.txt',
                                 dev_file="file_train.txt")

if __name__ == '__main__':
    doWork()