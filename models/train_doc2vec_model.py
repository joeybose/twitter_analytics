import os.path
import sys
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
import numpy as np

def to_taggeddoc(line_ls):
    return TaggedDocument(line_ls[1:], ['SENT' + line_ls[0]])

DOC_F='proc_dir/doc.txt'

with open(DOC_F) as f:
    lines = map(lambda x: to_taggeddoc(utils.to_unicode(x).strip().split()), f.readlines())
    
import gc; gc.collect()

model = Doc2Vec(min_count=1, window=8, size=100, sample=1e-4, negative=5, workers=10)
np.random.shuffle(lines)

model.build_vocab(lines)
print 'lines', len(lines)

for epoch in range(20):
    print 'epoch', epoch
    np.random.shuffle(lines)
    model.train(lines)
    #model.save('proc_dir/doc_model.d2v')
    
model.save('proc_dir/doc_model.d2v')
line_map={l.tags[0]:l.words for l in lines}
import joblib
joblib.dump(line_map,'proc_dir/line_map.pickle',compress=9)
import pdb; pdb.set_trace()




    
        
