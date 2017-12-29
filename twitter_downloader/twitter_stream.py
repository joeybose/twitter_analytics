from __future__ import absolute_import, print_function

import sys
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="uCZmfebaEpDK198eEGktCk3FM"
consumer_secret="Jykl9XH1h8ADx85SI4t5gXXO0gDmSRzq5X3ASUu9DTtwt8EYYk"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token="3341599034-SMqceazCjUivsydIhQaTbo17DoT9XskXJOnh4f1"
access_token_secret="qV7YkOSGdihV8lYSdkCcSmECHy120AcoJPE7jRqKQcAko"

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)

def print_usage():
    print("USAGE: python twitter_stream.py KW_FILE > stream_out.txt")

def stream(words):
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=words, languages=['en'])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    kw_fpath = sys.argv[1]
    
    with open(kw_fpath,'r') as f:
        words = map(lambda x: x.decode('utf-8').strip(), f.readlines())
    
    
    while True:
        try:
            stream(words)
        except Exception as err:
            sys.stderr.write("Caught err {}, reconneting to continue streaming".format(str(err)))
    
