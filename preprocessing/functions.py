import os
import re
import spacy
import string
import pymorphy2
import pandas as pd
from gensim.models import Phrases
from nltk.tokenize import word_tokenize
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex
from spacy.pipeline import merge_entities


# Do NOT split intra-hyphen words (spaCy)
def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search, \
                     suffix_search=suffix_re.search, \
                     infix_finditer=infix_re.finditer, \
                     token_match=None)

def read_texts(sources):
    dirpath = os.path.dirname(os.getcwd()) + '\\corpus\\'
    dfs = []
    for source in sources:
        df = pd.read_json(dirpath + source)
        df['Full text'] = df['Full text'].str.replace(r'[\xad…]', '', \
                                                      regex=True).astype('str') # soft hyphen
        df['Full text'] = df['Full text'].str.replace(r'[\xa0]', ' ', \
                                                      regex=True).astype('str') # no-break space
        df['Full text'] = df['Full text'].str.replace('ё', 'е').astype('str') # normalize spelling
        df = df[df['Author'] != 'N/A'] # remove texts without the author
        df = df[df['Topics'] != ''] # remove texts with no pre-defined topics
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_tokens(texts):
    return [word_tokenize(text) for text in texts]

def get_lemmas(texts):
    return [[morph.normal_forms(word)[0] if '_' not in word \
             else '_'.join(morph.normal_forms(i)[0] for i in word.split('_')) \
             for word in text] for text in texts]

def get_named_ents(texts):
    ner_types = ['ORG', 'PERSON', 'GPE', 'LOC']
    data = []
    for text in texts:
        data.append([str(word).replace(' ', '_') if word.ent_type_ and word.ent_type_ in ner_types \
                     else str(word) for word in nlp(text)])
    return data

def get_ngrams(texts, min_count, threshold):
    ngram = Phrases(texts, min_count=min_count, threshold=threshold)
    return ngram[texts]

def get_nouns_adj(texts):
    pos = ['NOUN', 'ADJF']
    return [[word for word in text if morph.parse(word)[0].tag.POS in pos] \
            for text in texts]

def clean_texts(texts):
    punct = string.punctuation # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    punct += '—“”«»<>…...°1234567890'
    remove = punct.replace('_', '').replace('-', '') # do not remove
    pattern = r"[{}]".format(remove)
    data = [[re.sub(pattern, '', word).rstrip('_-') for word in text] \
            for text in texts]
    # ignore words that contain only latin characters
    return [[re.sub('[Ёё]', 'е', word) for word in text if word and re.search(r'[^a-zA-Z]+', word)] \
            for text in data]

def remove_stopwords(texts):
    with open('swl_optimum.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    data = [[word for word in text if word not in stopwords] for text in texts]
    return [[word for word in text if word] for text in data]


# Load spaCy model
nlp = spacy.load('ru_core_news_lg')
nlp.tokenizer = custom_tokenizer(nlp)
nlp.add_pipe('merge_entities') # allows for merging with _
morph = pymorphy2.MorphAnalyzer()