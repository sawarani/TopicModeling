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
    dirpath = os.path.dirname(os.getcwd())
    dirpath += '\\corpus\\'
    js_docs = []
    for file in os.listdir(dirpath):
        if file.startswith('__'):
            with open(dirpath+file, 'r') as f:
                js = json.load(f)
                if len(js['Author']) > n: # no more than n texts per author
                    num = n
                else: 
                    num = len(js['Author'])
                add_texts_to_json(js, num, js_docs)
    return js_docs

def add_texts_to_json(json, n, texts):
    for i in range(n):
        text = json['Author'][i]['Full text']
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

def get_nouns(texts):
    return [[word for word in text \
             if morph.parse(word)[0].tag.POS == 'NOUN'] \
            for text in texts]

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

def get_corpus_size(n):
    texts = read_texts(n)
    tokens = [word_tokenize(doc) for doc in texts]
    tokens_clean = clean_texts(tokens)
    data = [[word for word in text if word] for text in tokens_clean]
    return sum([len(token) for token in data])

def count_tokens(texts):
    tokens = [word_tokenize(text) for text in texts]
    tokens_clean = clean_texts(tokens)
    data = [[word for word in text if word] for text in tokens_clean]
    count = sum([len(token) for token in data])
    return count, round(count/len(texts))  

def get_name_num(file, path, n):
    docs = []
    with open(path+file, 'r') as f:
        js = json.load(f)
        name = js['Author'][0]['Author']
        if len(js['Author']) > n: # no more than n texts per author
            num = n
        else:
            num = len(js['Author'])
        add_texts_to_json(js, num, docs)
        return docs, name, num

def get_stats(source, num):
    dirpath = os.path.dirname(os.getcwd())
    dirpath += '\\corpus\\'
    n, m, k = 0, 0, 0
    for file in os.listdir(dirpath):
        if file.endswith(f'{source}.json'):
            n += 1
            texts, name, num_art = get_name_num(file, dirpath, num)
            count, avg_count = count_tokens(texts)
            k += count
            m += num_art
            print(f'{name}:\n\tNumber of articles: {num_art}' + 
                f'\n\tTotal number of words: {count}' + 
                f'\n\tAverage number of words in an article: {avg_count}\n')
    print(f'Total number of authors: {n}' + 
        f'\nTotal number of articles: {m}' + 
        f'\nCorpus size (before preprocessing): {k:,}')


# Load spaCy model
nlp = spacy.load('ru_core_news_lg')
nlp.tokenizer = custom_tokenizer(nlp)
nlp.add_pipe('merge_entities') # allows for merging with _
morph = pymorphy2.MorphAnalyzer()