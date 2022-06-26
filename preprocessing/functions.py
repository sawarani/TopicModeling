import os
import re
import json
import spacy
import string
import pymorphy2
from gensim.models import Phrases
from nltk.tokenize import word_tokenize
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
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

def read_texts(n):
    dirpath = os.path.dirname(os.getcwd()) + '\\corpus\\'
    js_docs = []
    with open(dirpath+'elementy_texts.json', 'r') as f:
        js = json.load(f)
        for key in js:
            if len(key) > n: # no more than n texts per author
                num = n
            else:
                num = len(js[key])
            add_texts_to_json(js, num, key, js_docs)
    return js_docs

def add_texts_to_json(json, n, name, texts):
    for i in range(n):
        text = json[name][i]['Full text']
        no_hyphen = re.sub('[\xad…]', '', text) # soft hyphen character          
        no_space = re.sub('[\xa0]', ' ', no_hyphen) # no-break space
        texts.append(no_space)

def get_lemmas(texts):
    return [[morph.normal_forms(word)[0] if '_' not in word \
             else '_'.join(morph.normal_forms(i)[0] for i in word.split('_')) \
             for word in text] for text in texts]

def get_named_ents(texts):
    data = []
    for text in texts:
        data.append([str(word) if not word.ent_type_ else str(word).replace(' ', '_') \
                     for word in nlp(text)])
    return data

def get_ngrams(texts):
    ngram = Phrases(texts, min_count=10, threshold=100)
    return ngram[texts]

def get_nouns_adj(texts):
    return [[word for word in text \
             if morph.parse(word)[0].tag.POS == 'NOUN' \
             or morph.parse(word)[0].tag.POS == 'ADJF'] \
            for text in texts]

def clean_texts(texts):
    punct = string.punctuation # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    punct += '—“”«»<>…...°1234567890'
    remove = punct.replace('_', '').replace('-', '') # do not remove
    pattern = r"[{}]".format(remove)
    data = [[re.sub(pattern, '', word).strip('_-') for word in text] \
            for text in texts]
    return [[word.replace('__', '_') for word in text if word] \
            for text in data]

def remove_stopwords(texts):
    with open('swl_optimum.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
        data = [[word for word in text if re.sub('[Ёё]', 'е', word) \
                 not in stopwords or word not in stopwords] for text in texts]
        # ignore words that contain only latin characters
        return [[word for word in text if re.search(r'[^a-zA-Z]+', word)] \
                for text in data]

def get_corpus_size(texts):
    tokens = [word_tokenize(doc) for doc in texts]
    tokens_clean = clean_texts(tokens)
    data = [[word for word in text if word] for text in tokens_clean]
    return sum([len(token) for token in data])


# Load spaCy model
nlp = spacy.load('ru_core_news_lg')
nlp.tokenizer = custom_tokenizer(nlp)
nlp.add_pipe('merge_entities') # allows for merging with _
morph = pymorphy2.MorphAnalyzer()