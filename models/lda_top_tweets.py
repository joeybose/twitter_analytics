import joblib
import pdb
import pickle
import argparse
from gensim.models import LdaMulticore as LdaModel
from train_lda_model import show_lda_topics
import linecache
import os
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import make_corpus
from preprocess import prep_line
from gensim import corpora, models, similarities
from gensim import matutils
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
import pprint
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import operator
import re
import time
# Minimum line length for a document to be included in the corpus
MIN_LEN = 40

'''
Disclaimer This code doesnt run well yet!!
'''

def get_top_tweets(tweet_list, topicid, tweet_count=10):
    #pdb.set_trace()
    return [x for x in tweet_list if x[1][1][0] == topicid][0:tweet_count]

def scrub_tweets(tweet_list, word_bank, tweet_count=10):
    top_tweets = []
    for x in tweet_list:
#	pdb.set_trace()
	tokens = set(x[1][0].split())
	ii = len(tokens.intersection(word_bank)) > 0
	if ii:
	    top_tweets.append(x[1][0])
    return top_tweets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer topic distribution for tweets and return top tweets per topic')
    parser.add_argument('tweet_file', help=('twitter dump where each'
                                           ' line is a tweet thread.'))
    args = parser.parse_args()
    tweet_file = args.tweet_file
    dictionary = corpora.Dictionary.load('lda/samsung1.dict')
    lda = models.LdaModel.load('lda/samsung.lda')
    corpus = corpora.MmCorpus('lda/samsung1.mm')
   # print("Loaded Model")
    tweet_list = {}  
    start = time.time()
    results = lda.print_topics(num_topics=10, num_words=10)
    pprint.pprint(results)
    results = lda.show_topics(formatted=False)
    topic_words = {}
    for i in xrange(0,10):
        for j in xrange(0,10):
            word = results[i][1][j][0]
            topic_words.setdefault(i,[]).append(word)
    word_bank = [] 
    for i in topic_words:
	if i == 1 or i == 4 or i == 7:
	   for word in topic_words[i]:
		word_bank.append(word)
#    pdb.set_trace()

    with open(tweet_file, 'r') as f:
	for i, line in enumerate(f):
         #   if len(line) > MIN_LEN:
         #       tweet_id = str(line.split(',', 1)[0])
		#pdb.set_trace()
		l = re.sub("\n","",line)
		no_quote = re.sub("'","",l)
		doc_line = re.sub("\\n","", no_quote)
	#	line = re.sub("[","",line)
	#	line = re.sub("]","",line)
		#pdb.set_trace()
         #       doc_line = str(line.split(',', 1)[1])
                tweet_list.setdefault(i, []).append(doc_line)
                new_vec = dictionary.doc2bow(doc_line.lower().split())
                topic_dist = lda[new_vec]
                tweet_list.setdefault(i, []).append(max(topic_dist,key=lambda x: x[1]))
    
    filter_list = ['samsung','iphone','galaxy','fbi','apple']
    tweet_list = sorted(tweet_list.items(), key=lambda (k, v): (v[1], v[1][1]), reverse = True) 
    for i in xrange(0,10):
	top_tweets = get_top_tweets(tweet_list, i, 10)
	pprint.pprint(top_tweets)
    #pdb.set_trace()
    #top_tweets = scrub_tweets(tweet_list,filter_list,10)
    end = time.time()
    elapsed = end - start
    print("Time taken to read through file %d" %(elapsed))
    #pprint.pprint(top_tweets)
   # pprint.pprint(topic_words)

