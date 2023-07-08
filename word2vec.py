import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import os
if not os.path.exists('word2vec.pkl'):
    print('Downloading word2vec model...')
    # import requests
    # url = 'http://dihana.cps.unizar.es/~cadrete/models/word2vec.pkl'
    # r = requests.get(url, allow_redirects=True)    
    # open('word2vec.pkl', 'wb').write(r.content)
    os.system('wget http://dihana.cps.unizar.es/~cadrete/models/word2vec.pkl')

import numpy as np
class Word2Vec(object):
    def __init__(self):
        super().__init__()
        self.word2ind = {}
        self.ind2word = {}
        self.xn = []

    def __getitem__(self, word):
        if ' ' in word:
            words = word.split(' ')
            w = 0
            for word in words:
                w += self[word]
            return w / len(words)
        else:
            return self.xn[self.word2ind[word.lower()]]
    
    def __len__(self):
        return len(self.word2ind)
    
    def similarity(self, w, v):
        return np.dot( self.xn[self.word2ind[w]], self.xn[self.word2ind[v]] )

    def most_similar(self, pos=[], neg=[], n=1):   
        if type(pos) == str:
            pos = [pos]
        if type(neg) == str:
            neg = [neg]
        vp = [self.word2ind[w] for w in pos]
        vn = [self.word2ind[w] for w in neg]
        
        vxp = np.sum( [self.xn[w] for w in vp], axis=0 )
        vxn = np.sum( [self.xn[w] for w in vn], axis=0 )
        v = vxp - vxn
        v = v / np.linalg.norm(v)
        s = np.dot(self.xn, v)        
        ind = np.argsort(s)[::-1]        
        
        r = [(self.ind2word[i], float(s[i])) for i in ind[:n+len(pos)] if s[i] > 0 and not i in vp]
        return r[:n] 


import pickle
with open('word2vec.pkl', 'rb') as f:
    w2i, i2w, xn = pickle.load(f)

word2vec = Word2Vec()
word2vec.word2ind = w2i
word2vec.ind2word = i2w
word2vec.xn = xn

print( word2vec['Mejorar la calidad del vending'])


