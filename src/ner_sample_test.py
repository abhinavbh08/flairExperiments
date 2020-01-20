from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy.gold import biluo_tags_from_offsets
import re
from spacy.tokenizer import Tokenizer

from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


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
    def custom_tokenizer(nlp):
        infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        # suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
        suffix_re = re.compile(
            '…$|……$|,$|:$|;$|\\!$|\\?$|¿$|؟$|¡$|\\($|\\)$|\\[$|\\]$|\\{$|\\}$|<$|>$|_$|#$|\\*$|&$|。$|？$|！$|，$|、$|；$|：$|～$|·$|।$|،$|۔$|؛$|٪$|\\.\\.+$|…$|\\\'$|"$|”$|“$|`$|‘$|´$|’$|‚$|,$|„$|»$|«$|「$|」$|『$|』$|（$|）$|)
        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=None)


    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = custom_tokenizer(nlp)

    doc = nlp(
        u'Note: Since the fourteenth century the practice of “medicine” has become a profession; and more importantly, it\'s a male-dominated profession.')
    print([token.text for token in doc])
    # check_ner()
