from gensim import corpora, models, similarities
import time
import numpy as np
from gensim.models import LdaMulticore as LdaModel
import argparse

if __name__ == '__main__':
    lda_filename    = 'lda/samsung.lda'
    corpus = corpora.MmCorpus('lda/samsung1.mm')
    dictionary = corpora.Dictionary.load('lda/samsung1.dict')
    start = time.time()
    lda = LdaModel(corpus, num_topics=10,
                   alpha=1./100, eta=.2, chunksize=10000,
                   workers=11, passes=10, decay=0.75,
                   id2word=dictionary)

    print('Saving model')
    end = time.time()
    elapsed = end - start
    print "Time taken for LDA training: ", elapsed, "seconds."
    lda.print_topics()
    lda.save(lda_filename)
    print("lda saved in %s " % lda_filename)
