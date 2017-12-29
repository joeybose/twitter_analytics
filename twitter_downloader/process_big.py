import argparse
import json
from itertools import chain
import nltk
import re
import traceback
import os
import warnings
import shutil
import linecache


# Tokenization function to use to break up the text into words
tokenize = nltk.wordpunct_tokenize


def include_tweet(cleaned_text, exclude_terms=None, include_terms=None):
    ie = False
    ii = True

    if exclude_terms or include_terms:
        tokens = set(cleaned_text.split())

    if exclude_terms:
        ie = len(tokens.intersection(exclude_terms)) > 0

    if include_terms:
        ii = len(tokens.intersection(include_terms)) > 0

    return (not ie) or ii


def process_twt(twt_json):
    text = twt_json['text']

    # remove urls and user mentions
    entries = twt_json['entities']
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    urls = (e["url"] for e in entries["urls"])
    users = ("@"+e["screen_name"] for e in entries["user_mentions"])
    text = reduce(lambda t, s: t.replace(s, ""), chain(urls, users), text)
    # remove html entities
    text = re.sub(r"&\w*;", "", text)

    # remove the retweet prefix
    text = text[2:] if text.startswith('RT') else text

    # tokenize into words
    text = tokenize(text)

    # restrict to word that do not contain non-alphabetical characters  #TODO
    sentence = ' '.join([w.lower() for w in text if w.isalpha()])

    return sentence

def dump_to_file(unparsed_file,out_dir, filename, tweet_list):
         for key, values in tweet_list:
             doc_line = ""
             first_arg = True
             #print(values)
             with open(os.path.join(out_dir, unparsed_file), 'r') as w, \
                  open(os.path.join(out_dir, filename), 'a') as f:
                 for value in values:
                    #print(value)
                    if first_arg == True:             #print value
                        doc_line = (linecache.getline(os.path.join(out_dir, unparsed_file),int(value+1))).rstrip('\n')
                        first_arg = False
                    else:
                        line = (linecache.getline(os.path.join(out_dir, unparsed_file),int(value+1))).rstrip('\n')
                        doc_line = doc_line.rstrip('\n') + " "+ str(line.split(',', 1)[1])
                        print(doc_line)
                 f.write('{cleaned_text}\n'.format(cleaned_text=doc_line))


                 #print(line)




def process(twt_dump_f, out_dir, cleaned_txt_out, ii_file=None, ie_file=None, keep_duplicates=False):
    II = set(open(ii_file).read().split()) if ii_file else None
    IE = set(open(ie_file).read().split()) if ie_file else None

    if IE and II:
        IE = IE.difference(II)

    tweet_set = set([])
    tweet_list = {}
    line_counter = 0
    with open(os.path.join(out_dir, cleaned_txt_out), 'w') as w, \
         open(twt_dump_f,'r') as f:

        for i, line in enumerate(f):
            cleaned_line = line.strip()
            id_str = None
            reply_exists = False

            if cleaned_line:
                try:
                    twt_json = json.loads(cleaned_line)
                    if 'text' in twt_json:
                        cleaned_text = process_twt(twt_json)
                        twt_json['cleaned_text'] = cleaned_text
                        id_str = str(twt_json['id_str'])
                        if twt_json['in_reply_to_status_id'] is not None:
                            reply_id = str(twt_json['in_reply_to_status_id'])
                        else:
                            reply_id = None
                        # check if the tweet should be included in the dataset
                        # based on included/excluded terms and the duplication setting
                        if ((keep_duplicates or cleaned_text not in tweet_set) and
                            include_tweet(cleaned_text, IE, II)):

                        #    with open(os.path.join(out_dir, '{tweet_id}.json'.format(tweet_id=id_str)), 'w') as f:
                        #        f.write(json.dumps(twt_json))

                            if reply_id is not None:
                                tweet_list.setdefault(reply_id, []).append(line_counter)
                                w.write('{tweet_id},{cleaned_text}\n'.format(tweet_id=reply_id, cleaned_text=cleaned_text.encode('UTF-8')))
                            else:
                                tweet_list.setdefault(id_str, []).append(line_counter)
                                w.write('{tweet_id},{cleaned_text}\n'.format(tweet_id=id_str, cleaned_text=cleaned_text.encode('UTF-8')))

                            if not keep_duplicates:
                                tweet_set.add(cleaned_text)
                            #reply_id = None
                            line_counter = line_counter + 1
                except Exception:
                    tb = traceback.format_exc()
                    warnings.warn("Failed to parse tweet at line {i}, id {id_str}: \n {tb}".format(
                        i=i, id_str=id_str, tb=tb))

                if i % 10000 == 0:
                    print 'Processed {} tweets'.format(i)
                    print (len(tweet_list))
        w.close()
        my_dict = sorted(tweet_list.items(), key=lambda x: x[1])
        #print(my_dict)
        dump_to_file(cleaned_txt_out, out_dir, 'parsed.txt',my_dict)


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

    parser.add_argument('--cleaned_txt_out', nargs='?', default='big.txt')
    parser.add_argument('--include_wd_file', nargs='?', default=None)
    parser.add_argument('--exclude_wd_file', nargs='?', default=None)
    parser.add_argument('--keep_duplicates', default=False, action='store_true')

    args = parser.parse_args()

    dump_file = args.dump_file
    out_dir = args.out_dir
    cleaned_txt_out = args.cleaned_txt_out

    ii_file = args.include_wd_file
    ie_file = args.exclude_wd_file

    process(dump_file, out_dir, cleaned_txt_out, ii_file, ie_file)
