from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy.gold import biluo_tags_from_offsets

nlp = spacy.load('en_core_web_sm')

def getNewTags(tags):
    new_tags = []
    for tag in tags:
        if tag[0] == "U":
            new_tags.append("B" + tag[1:])
        elif tag[0] == "L":
            new_tags.append("I" + tag[1:])
        else:
            new_tags.append(tag)

    return new_tags

def writeDataToTextFile(docs):
    with open("data/file.txt", "w", encoding="utf-8") as file:
        for doc in docs:
            for name, label in zip(doc[0], doc[1]):
                print(name, label)
                file.write(name + " " + label + "\n")
            file.write("\n")

def convertDataToFlair(TRAIN_DATA):

    print("HELLOWORLD")
    docs = []
    for text, annot in TRAIN_DATA:
        tokens = []
        doc = nlp(text)
        tags = biluo_tags_from_offsets(doc, annot['entities'])
        for token in doc:
            tokens.append(token.text)
        tags = getNewTags(tags)
        docs.append((tokens, tags))
        # then convert L->I and U->B to have IOB tags for the tokens in the doc

    writeDataToTextFile(docs)
    docs