import numpy as np
from sklearn.manifold import MDS
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

start = time.time() # Start ti
model = Word2Vec.load("lda/word_model.w2v")
word_vectors = model.syn0
print word_vectors.shape
print word_vectors.shape[0]
print word_vectors.shape[1]
#svd = TruncatedSVD(n_components=10, random_state=0)
#svd_tfidf = svd.fit_transform(word_vectors)
MDS_model = MDS(n_components=2, n_jobs=-2,random_state=0)
np.set_printoptions(suppress=True)
MDS_model.fit(word_vectors)
s = joblib.dump(MDS_model, 'MDS_model.pkl')
end = time.time()
elapsed = end - start
print "Time taken for MDS clustering: ", elapsed, "seconds."
