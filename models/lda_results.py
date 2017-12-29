import joblib
import pickle
import argparse
from gensim.models import LdaMulticore as LdaModel
from train_lda_model import show_lda_topics


if __name__ == '__main__':
    lda_big = joblib.load("lda/lda_40.pickle")
 

   
    print("Results for tweet threads")
    results = show_lda_topics(lda_big["lda"],40,10)
