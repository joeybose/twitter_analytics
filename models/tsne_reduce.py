import numpy as np
from sklearn.manifold import TSNE
import joblib
import pickle
from sklearn.externals import joblib
import time
from sklearn.cluster import MiniBatchKMeans
import argparse
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import os.path
import sys
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

if __name__ == '__main__':
    print("Starting")
    start = time.time() # Start ti
    model = Word2Vec.load("/home/joey/lab-twitter-analytics/models/lda/azure_w2v/word2vec/word_model.w2v")
    word_vectors = model.syn0
    print word_vectors.shape
    w2v_norm = word_vectors / np.linalg.norm(word_vectors)
    svd = TruncatedSVD(n_components=50, random_state=0)
    svd_reduce = svd.fit_transform(w2v_norm)
    tsne_model = TSNE(n_components=10,init="pca",random_state=0)
    np.set_printoptions(suppress=True)
    tsne_model.fit(svd_reduce)
    s = joblib.dump(tsne_model, 'tsne_model_new.pkl')
    end = time.time()
    elapsed = end - start
    print "Time taken for tsne clustering: ", elapsed, "seconds."
