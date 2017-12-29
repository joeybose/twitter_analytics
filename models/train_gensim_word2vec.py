import os.path
import sys
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Word2Vec
import numpy as np
import argparse
import nltk
import re
import random
import time
from collections import defaultdict
from configparser import ConfigParser
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import utils
from string import digits
import argparse
import os
from pprint import pprint
from nltk.tokenize import RegexpTokenizer

def make_taggeddoc(line_ls, tag):
    return TaggedDocument(line_ls,['SENT' + str(tag)])

def to_taggeddoc(line_ls):
    return TaggedDocument(line_ls[1:], ['SENT' + line_ls[0]])

file_path = os.path.dirname(__file__)

# Load included and excluded words
with open(os.path.join(file_path, 'vocabs/extra_stopwords.txt'), 'r') as f:
    _extra_stops = set(map(str.strip, f.readlines()))

default_exclude_list = set(stopwords.words('english'))
default_exclude_list.update(_extra_stops)

with open(os.path.join(file_path, 'vocabs/v2w_vocab.txt'), 'r') as f:
    default_include_list = set(map(str.strip, f.readlines()))


MIN_LEN = 80

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
    return words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer topic distribution for tweets and return top tweets per topic')
    parser.add_argument('tweet_file', help=('twitter dump where each'
                                           ' line is a tweet thread.'))
    args = parser.parse_args()
    DOC_F=args.tweet_file

    documents = [
        prep_line(line, default_include_list, default_exclude_list)
        for line in prep_lines(args.tweet_file, MIN_LEN, True)
    ]
    #lines = [make_taggeddoc(tweet, i) for i, tweet in enumerate(documents)]
    import gc; gc.collect()
    print "Training model..."

    model = Word2Vec(min_count=1, window=8, size=50, sample=1e-4, negative=5, workers=3)
    np.random.shuffle(documents)

    model.build_vocab(documents)
    print 'lines', len(documents)

    for epoch in range(20):
        print 'epoch', epoch
        np.random.shuffle(documents)
        model.train(documents)

    model.save('lda/word_model.w2v')
