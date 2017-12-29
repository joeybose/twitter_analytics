'''
This script
* Loads documents as aggregation of tweets stored in a MongoDB collection
* Cleans up the documents
* Creates a dictionary and corpus that can be used to train an LDA model
* Training of the LDA model is not included but follows:
  lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=100)
Author: Alex Perrier
Python 2.7
'''

import langid
import nltk
import re
import random
import time
from collections import defaultdict
from configparser import ConfigParser
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim import utils
from string import digits
import argparse
import os
from pprint import pprint
from nltk.tokenize import RegexpTokenizer

file_path = os.path.dirname(__file__)

# Load included and excluded words
with open(os.path.join(file_path, 'vocabs/extra_stopwords.txt'), 'r') as f:
    _extra_stops = set(map(str.strip, f.readlines()))

default_exclude_list = set(stopwords.words('english'))
default_exclude_list.update(_extra_stops)

with open(os.path.join(file_path, 'vocabs/v2w_vocab.txt'), 'r') as f:
    default_include_list = set(map(str.strip, f.readlines()))

lm = WordNetLemmatizer()

# Lemmatizer to collapse word forms into one

# Minimum line length for a document to be included in the corpus
MIN_LEN = 80


def filter_lang(lang, documents):
    doclang = [  langid.classify(doc) for doc in documents ]
    return [documents[k] for k in range(len(documents)) if doclang[k][0] == lang]

def prep_lines(txt_file, min_len, randomize):
    with open(txt_file) as f:
        lines = []
        for line in f:
            _, text = line.split(',', 1)
            if len(text) > min_len:
                lines.append(utils.to_unicode(text.lower()).strip())
    if randomize:
        random.shuffle(lines)
    return lines


def prep_line(line, include_list, exclude_list):
    line = line.lower()
    words = line.split()
    if include_list:
        words = [w for w in words if w in include_list]
    if exclude_list:
        words = [w for w in words if w not in exclude_list]

    lemmatized = []
    for word in words:
        for pos in [wn.NOUN, wn.VERB]:
            l = lm.lemmatize(word, pos)
            if l != word:
                lemmatized.append(l)
                break
        else:
            lemmatized.append(word)
    words = lemmatized
    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LDA model')
    parser.add_argument('tweet_file', help=('path to twitter downloader dump where each line is a cleaned tweet'))
    parser.add_argument('--include_list', nargs='?', default=None)  # see preprocess for default list
    parser.add_argument('--exclude_list', nargs='?', default=None)  # see preprocess for default list
    args = parser.parse_args()

    documents = [
        prep_line(line, default_include_list, default_exclude_list)
        for line in prep_lines(args.tweet_file, MIN_LEN, True)
    ]

    # Build a dictionary where for each document each word has its own id
    dictionary = corpora.Dictionary(documents)
    dictionary.compactify()
    # and save the dictionary for future use
    dictionary.save('lda/lda_middle_east.dict')

    # We now have a dictionary with 26652 unique tokens

    # Build the corpus: vectors with occurence of each word for each document
    # convert tokenized documents to vectors
    corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format

    corpora.MmCorpus.serialize('lda/lda_middle_east.mm', corpus)
# this corpus can be loaded with corpus = corpora.MmCorpus('alexip_followers.mm')
