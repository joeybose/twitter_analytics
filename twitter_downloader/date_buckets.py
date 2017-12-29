import argparse
import json
from itertools import chain
import nltk
import re
import traceback
import os
import warnings
import linecache
import time
from datetime import datetime

def process(twt_dump_f, out_dir):
    tweet_set = set([])
    tweet_list = {}
    with open(twt_dump_f,'r') as f:
        for i, line in enumerate(f):
            cleaned_line = line.strip()
            if cleaned_line:
                try:
                    twt_json = json.loads(cleaned_line)
                    if 'text' in twt_json:
                        date = str(twt_json['created_at'])
                        dt = time.strptime(date, '%a %b %d %H:%M:%S +0000 %Y')
                        tweet_date = time.strftime('%b%d',dt)
                        #print "Date is : %s " % tweet_date
                        filename = tweet_date+ '.txt'
                        with open(os.path.join(out_dir, filename), 'a+') as temp_file:
                            temp_file.write(line)
                        temp_file.close
                except Exception:
                    tb = traceback.format_exc()
                    warnings.warn("Failed to parse tweet at line {i}: \n {tb}".format(
                        i=i, tb=tb))
                if i % 10000 == 0:
                    print 'Processed {} tweets'.format(i)
        w.close()
        print("Total Number of Tweets is: ")
        print (len(tweet_list))

if __name__ == '__main__':
    """
    Process twitter downloader dump.
    """
    parser = argparse.ArgumentParser(description='Process twitter downloader dump.')
    parser.add_argument('dump_file', help=('twitter downloader dump where each'
                                           ' line is a tweet in json format.'))

    parser.add_argument('out_dir', help=('output directory path where parsed'
                                         ' tweets in json and cleaned text '
                                         'file will be stored.'))

    args = parser.parse_args()
    dump_file = args.dump_file
    out_dir = args.out_dir
    process(dump_file, out_dir)
