from gensim import corpora, models, similarities
from time import time
import numpy as np
from gensim.models import LdaMulticore as LdaModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LDA model')
    parser.add_argument('tweet_file', help=('path to twitter downloader dump where each line is a cleaned tweet'))
    parser.add_argument('out_dir', help=('path output file to save the model'))
    parser.add_argument('--include_list', nargs='?', default=None)  # see preprocess for default list
    parser.add_argument('--exclude_list', nargs='?', default=None)  # see preprocess for default list

    parser.add_argument('num_topics', type=int)
    parser.add_argument('--npasses', type=int, default=50)
    parser.add_argument('--decay', type=float, default=.5)
    parser.add_argument('--chunksize', type=int, default=2000)

    lda_filename    = 'lda/middle_east_100.lda'
    args = parser.parse_args()
    corpus = corpora.MmCorpus('lda/lda_middle_east.mm')
    dictionary = corpora.Dictionary.load('lda/lda_middle_east.dict')

    lda = LdaModel(corpus, num_topics=100,
                   alpha=1./100, eta=.2, chunksize=10000,
                   workers=5, passes=100, decay=0.75,
                   id2word=dictionary)

    print('Saving model')
    lda.print_topics()
    lda.save(lda_filename)
    print("lda saved in %s " % lda_filename)
