from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy.gold import biluo_tags_from_offsets


def check_ner():
    tagger = SequenceTagger.load('ner')
    sentence = Sentence('I love Berlin!')
    tagger.predict(sentence)
    print(sentence.to_tagged_string())


    TRAIN_DATA = [
        ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
        ("I like London and Berlin.", {"entities": [(7, 13, "LOCSEX"), (18, 24, "LOCSEX")]}),
    ]

    nlp = spacy.load('en_core_web_sm')
    docs = []
    for text, annot in TRAIN_DATA:
        doc = nlp(text)
        tags = biluo_tags_from_offsets(doc, annot['entities'])
        print("TAGS->>>>>>>>>>>..", tags)

if __name__ == '__main__':
    check_ner()