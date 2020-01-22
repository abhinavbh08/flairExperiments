import pandas as pd
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


def convertDataToLstm(DATA, SLOTS_INFO, IDS, train):
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
        doc_things = []
        tokens = []
        doc = nlp(text)
        tags = biluo_tags_from_offsets(doc, annot['entities'])
        tags = getNewTags(tags)
        for i, tag in enumerate(tags):
            if (tag == "-"):
                for slot in SLOTS_INFO[j]:
                    if slot["slotValue"] in tokens[i]:
                        tags[i] = slot["slotName"]
                        break
        for i, token in enumerate(doc):
            doc_things.append((token.text, token.pos_, tags[i]))

        docs.append(doc_things)

    print(docs)