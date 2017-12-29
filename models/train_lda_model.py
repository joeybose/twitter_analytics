import argparse
import joblib

from gensim.models import LdaMulticore as LdaModel

from preprocess import make_corpus


def fit_lda(corpus, id2word, num_topics=100, npasses=200, decay=.75, chunksize=10000):
    print 'Fitting LDA'
    lda = LdaModel(corpus, num_topics=num_topics,
                   alpha=0.001, eta=.2, chunksize=chunksize,
                   workers=5, passes=npasses, decay=decay,
                   id2word=id2word)
    print('Done')
    return lda


def show_lda_topics(lda, num_topics=100, num_words=10):
    topics = lda.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    for ti, topic in topics:
        print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[0], t[1]) for t in topic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LDA model')
    parser.add_argument('tweet_file', help=('path to twitter downloader dump where each line is a cleaned tweet'))
    parser.add_argument('out_dir', help=('path output file to save the model'))
    parser.add_argument('--include_list', nargs='?', default=None)  # see preprocess for default list
    parser.add_argument('--exclude_list', nargs='?', default=None)  # see preprocess for default list

    parser.add_argument('num_topics', type=int, default=100)
    parser.add_argument('--npasses', type=int, default=200)
    parser.add_argument('--decay', type=float, default=.5)
    parser.add_argument('--chunksize', type=int, default=10000)

    args = parser.parse_args()

    corpus, vec, id2word, vocab = make_corpus(args.tweet_file, args.include_list, args.exclude_list)
    lda = fit_lda(corpus, id2word,
                  num_topics=args.num_topics,
                  npasses=args.npasses, decay=args.decay, chunksize=args.chunksize)
    print('Saving model')
    joblib.dump({'lda': lda, 'vec': vec, 'id2word': id2word, 'vocab': vocab},args.out_dir , compress=9)

