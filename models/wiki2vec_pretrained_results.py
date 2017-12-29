from gensim.models import Word2Vec

if __name__ == '__main__':
    from gensim.models import Word2Vec
    model = Word2Vec.load("wiki_model/wiki.en.word2vec.model")
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model.doesnt_match("breakfast cereal dinner lunch".split()))
    print(model.similarity('woman', 'man'))
