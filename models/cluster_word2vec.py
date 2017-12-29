import joblib
import pickle
from sklearn.externals import joblib
import time
from sklearn.cluster import MiniBatchKMeans
import argparse
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Word2Vec

if __name__ == '__main__':
    model = Word2Vec.load("en_1000_no_stem2/en.model")

    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 20
    print("Vocab size is %s", word_vectors.shape[0])

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = MiniBatchKMeans( n_clusters = num_clusters, init = "random" )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    s = joblib.dump(idx, 'Word2Vec_clusters.pkl')

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.index2word, idx ))

    # For the first 10 clusters
    for cluster in xrange(0,10):
        #
        # Print the cluster number
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words
