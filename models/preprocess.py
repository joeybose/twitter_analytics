import os
import random

from gensim import matutils
from gensim import utils
from gensim.models.doc2vec import TaggedDocument

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

file_path = os.path.dirname(__file__)

# Load included and excluded words
with open(os.path.join(file_path, 'vocabs/extra_stopwords.txt'), 'r') as f:
    _extra_stops = set(map(str.strip, f.readlines()))

default_exclude_list = set(stopwords.words('english'))
default_exclude_list.update(_extra_stops)

with open(os.path.join(file_path, 'vocabs/v2w_vocab.txt'), 'r') as f:
    default_include_list = set(map(str.strip, f.readlines()))

# Lemmatizer to collapse word forms into one
lm = WordNetLemmatizer()

# Minimum line length for a document to be included in the corpus
MIN_LEN = 80


def remove_wds(wl, exclude, include=None):
    if include:
        return [w for w in wl if (w not in exclude and w in include)]
    else:
        return [w for w in wl if w not in exclude]


def prep_lines(txt_file, min_len, randomize):
    with open(txt_file) as f:
        lines = []
        for line in f:
            _, text = line.split(',', 1)
            if len(text) > min_len:
                lines.append(utils.to_unicode(text).strip())
    if randomize:
        random.shuffle(lines)
    return lines


def prep_line(line, include_list, exclude_list):
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


def make_corpus(txt_file, include_list, exclude_list):
    if exclude_list is None:
        exclude_list = default_exclude_list
    if include_list is None:
        include_list = default_include_list

    lines = [
        prep_line(line, include_list, exclude_list)
        for line in prep_lines(txt_file, MIN_LEN,  True)
    ]

    vec = CountVectorizer(min_df=1, stop_words='english')
    X = vec.fit_transform(' '.join(l) for l in lines)
    vocab = vec.get_feature_names()
    id2word = dict([(i, s) for i, s in enumerate(vocab)])
    corpus = matutils.Sparse2Corpus(X.T)

    return corpus, vec, id2word, vocab
