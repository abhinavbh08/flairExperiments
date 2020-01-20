import spacy
from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer
import re

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


def writeDataToTextFile(docs, train):
    if train:
        name = "train_IOB_Format_file"
    else:
        name = "dev_IOB_format_file"
    with open("data/{}.txt".format(name), "w", encoding="utf-8") as file:
        for doc in docs:
            for name, label in zip(doc[0], doc[1]):
                print(name, label)
                file.write(name + " " + label + "\n")
            file.write("\n")


def convertDataToFlair(DATA, SLOTS_INFO, train):
    prefix_re = re.compile(r'''^[[("']''')
    suffix_re = re.compile(r'''[])"']$''')
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')

    # simple_url_re = re.compile(r'''[a-zA-Z0-9]/+''')

    def create_tokenizer(nlp):
        return Tokenizer(nlp.vocab,
                         rules={},
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer
                         )

    nlp.tokenizer = create_tokenizer(nlp)

    docs = []
    for j, (text, annot) in enumerate(DATA):
        tokens = []
        doc = nlp(text)
        tags = biluo_tags_from_offsets(doc, annot['entities'])
        for token in doc:
            tokens.append(token.text)
        tags = getNewTags(tags)
        for i, tag in enumerate(tags):
            if (tag == "-"):
                for slot in SLOTS_INFO[j]:
                    if slot["slotValue"] in tokens[i]:
                        tags[i] = slot["slotName"]
                        break
        docs.append((tokens, tags))
        # then convert L->I and U->B to have IOB tags for the tokens in the doc

    writeDataToTextFile(docs, train)
