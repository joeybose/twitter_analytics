import sys, os
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
import numpy as np
from gensim.matutils import argsort

from nltk.corpus import stopwords

def remove_wds(wl, exclude, include):
    return [w for w in wl if (w not in exclude and w in include)]

def mean_top(w2v, ws, ref_vecs, topn=2):
    inds = [w2v.vocab[w].index for w in ws]
    vecs = w2v.syn0norm[inds]
    max_sims = np.dot(vecs, ref_vecs.T).max(axis=1)
    rep = vecs[argsort(max_sims, topn=topn, reverse=True)].mean(axis=0)
    assert not np.isnan(rep).any()
    return rep
    
stopwords = set(stopwords.words('english'))
w2v=gensim.models.Word2Vec.load('wiki.en.word2vec.model')
w2v.init_sims(replace=False)

DOC_F='proc_dir/doc.txt'

with open(DOC_F) as f:
    lines = map(lambda x: (x[0],remove_wds(utils.to_unicode(x[1]).strip().split(), stopwords, w2v.vocab)),
                enumerate(f.readlines()))

II=set(open('hc_word_picked.txt').read().split())
inds = [w2v.vocab[w].index for w in II]
II_vec = w2v.syn0norm[inds]

import gc; gc.collect()
lines = [l for l in lines if l[1]]
import gc; gc.collect()
lines = [(l[0], l[1], mean_top(w2v, l[1], II_vec)) for l in lines]
import joblib
joblib.dump(lines, 'proc_dir/wbow_lines.pickle', compress=9)
import pdb; pdb.set_trace()
