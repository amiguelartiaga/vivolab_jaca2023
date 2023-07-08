import numpy as np
import requests
import justext as jt
import re, os
import pickle, gzip
import torch
from transformers import BertForMaskedLM, BertTokenizer

def imprimir_comienzo(texto):
    texto = re.sub("\n",' ', texto)
    x = [x for x in texto.split('.')[:3]]
    txt = '. '.join(x) + '.'
    txt = re.sub(' +', ' ', txt)
    print(txt)        

def limpiar_texto_web(text):
    stop = jt.get_stoplist("Spanish")
    out = [x.text for x in jt.justext(text,stop) if not x.is_boilerplate]
    return "\n".join(out)

def descargar_web(web):
    return requests.get(web).text

def descargar_lista_webs(webs):
    textos = [descargar_web(web) for web in webs]
    textos = [limpiar_texto_web(texto) for texto in textos]
    return textos

import unicodedata
def eliminar_acentos(x):    
    try:
        x = unicode(x, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    x = unicodedata.normalize('NFD', x)
    x = x.encode('ascii', 'ignore')
    x = x.decode("utf-8")
    return x

import re
from nltk.corpus import stopwords
def limpiar_texto(txt):
    txt = eliminar_acentos(txt)
    txt = txt.lower()
    txt = re.sub("[^a-z]",' ', txt)
    txt = [ w for w in txt.split(' ') if w not in stopwords.words('spanish') ]
    txt = [ w for w in txt if len(w) > 1]
    txt = " ".join(txt)
    txt = re.sub(' +', ' ', txt)
    return txt

import pandas as pd
def tabla_datos(x, cols, y):
    df = pd.DataFrame(x, columns=cols)
    df['etiqueta'] = y
    return df

# --------------------------------------------------------------------------------
def leer_w2vec(fichero='w2vec.gz'):
    if not os.path.exists(fichero):
        os.system('wget http://dihana.cps.unizar.es/~cadrete/models/w2vec.gz -O w2vec.gz')
 
    with gzip.open(fichero, 'rb') as f:
        x, w_ = pickle.load(f)
    return {w: x[i] for i, w in enumerate(w_)}

def leer_bert():
    if not os.path.exists('pytorch/pytorch_model.bin'):
        os.system('wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz')
        os.system('wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/vocab.txt')
        os.system('wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/config.json')
        os.system('tar -xzvf pytorch_weights.tar.gz')
        os.system('mv config.json pytorch/.')
        os.system('mv vocab.txt pytorch/.')
    tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
    bert = BertForMaskedLM.from_pretrained("pytorch/")
    bert.eval()
    return tokenizer, bert

# --------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def entrenar_clasificador1(train, labels):
    bow = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = bow.fit_transform(train)
    model = MultinomialNB().fit(x, labels)
    return bow, model
    
def evaluar_clasificador1(bow, model, texto):
    x = bow.transform([texto])
    print('predicción clasificador1: categoría %d' % model.predict(x)[0])

def transformar1(bow, train, labels):
    train = bow.transform(train).toarray()
    return tabla_datos(train, bow.get_feature_names_out(), labels)


def entrenar_clasificador2(train, labels):
    bow = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = bow.fit_transform(train)
    tfidf = TfidfTransformer()
    x = tfidf.fit_transform(x)
    model = MultinomialNB().fit(x, labels)
    return bow, tfidf, model
    
def evaluar_clasificador2(bow, tfidf, model, texto):
    x = bow.transform([texto])
    x = tfidf.transform(x)
    print('predicción clasificador2: categoría %d' % model.predict(x)[0])

def transformar2(bow, tfidf, train, labels):
    train = tfidf.transform(bow.transform(train)).toarray()
    return tabla_datos(train, bow.get_feature_names_out(), labels)

def word2vec_linea(word2vec, line):
    x = [word2vec[x] for x in line.split() if x in word2vec] 
    return sum(x)/len(x)

def entrenar_clasificador3(word2vec, train, labels):
    train = [ limpiar_texto(x) for x in train ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)
    model = LogisticRegression().fit(x, labels)
    return model

def evaluar_clasificador3(word2vec, model, texto):
    train = [ limpiar_texto(texto) ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)
    print('predicción clasificador3: categoría %d' % model.predict(x)[0])


def extraer_bert(bert, tokenizer, text):
    text = "[CLS] "+ text+" [SEP]" 
    tokens = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    x = bert(tokens_tensor[:,:500], output_hidden_states=True)
    return x.hidden_states[-1][0].detach().numpy()

def extraer_bert1(bert, tokenizer, text):
    text = "[CLS] "+ text+" [SEP]" 
    tokens = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    x = bert(tokens_tensor[:,:500], output_hidden_states=True)
    return x.hidden_states[-1][0, 0].detach().numpy()

def extraer_bert2(bert, tokenizer, text):
    text = "[CLS] "+ text+" [SEP]" 
    tokens = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    x = bert(tokens_tensor[:,:500], output_hidden_states=True)
    return x.hidden_states[-1][0].mean(0).detach().numpy()

def entrenar_clasificador4(bert, tokenizer, train, labels):
    train = [ extraer_bert2(bert, tokenizer, x) for x in train ]
    x = np.stack(train)
    model = LogisticRegression().fit(x, labels)
    return model

def evaluar_clasificador4(bert, tokenizer, model, texto):
    train = [ extraer_bert2(bert, tokenizer, texto)  ]    
    x = np.stack(train)
    print('predicción clasificador4: categoría %d' % model.predict(x)[0])

# ----------------------------------------------------------------------------------

from sklearn.cluster import KMeans
def clustering1(word2vec, textos, numero_clusters):
    textos = [ limpiar_texto(x) for x in textos ]
    textos = [ word2vec_linea(word2vec,linea) for linea in textos]
    x = np.array(textos)
    kmeans = KMeans(n_clusters=numero_clusters, random_state=0).fit(x)
    return kmeans.labels_

from sklearn.cluster import SpectralClustering
def clustering2(word2vec, textos, numero_clusters):
    textos = [ limpiar_texto(x) for x in textos ]
    textos = [ word2vec_linea(word2vec,linea) for linea in textos]
    x = np.array(textos)
    cl = SpectralClustering(assign_labels='discretize', n_clusters=numero_clusters,random_state=0).fit(x)
    return cl.labels_

def clustering3(bert, tokenizer, textos, numero_clusters):
    train = [ extraer_bert2(bert, tokenizer, x) for x in textos ]
    x = np.stack(train)
    kmeans = KMeans(n_clusters=numero_clusters, random_state=0).fit(x)
    return kmeans.labels_

# ----------------------------------------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity
def preparar_buscador1(train):
    bow = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = bow.fit_transform(train)
    tfidf = TfidfTransformer()
    x = tfidf.fit_transform(x)
    return bow, tfidf, x

def consulta_buscador1(bow, tfidf, xdb, textosdb, texto, topk=3):
    x = bow.transform([texto])
    x = tfidf.transform(x)
    s = cosine_similarity(xdb, x).reshape(-1)
    ind = s.argsort()[-topk:][::-1]
    print('\nconsulta: ', texto)
    for n, i in enumerate(ind):
        print("(%d/%d) %f: " % (n+1, len(ind), s[i]), end='')
        imprimir_comienzo(textosdb[i])
        
def preparar_buscador2(word2vec, train):
    train = [ limpiar_texto(x) for x in train ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)    
    return x

def consulta_buscador2(word2vec, xdb, textosdb, texto, topk=3):
    x = [ limpiar_texto(texto) ]
    x = [ word2vec_linea(word2vec,linea) for linea in x]
    x = np.array(x)
   
    s = cosine_similarity(xdb, x).reshape(-1)
    ind = s.argsort()[-topk:][::-1]
    print('\nconsulta: ', texto)
    for n, i in enumerate(ind):
        print("(%d/%d) %f: " % (n+1, len(ind), s[i]), end='')
        imprimir_comienzo(textosdb[i])
        

def ver_similitud(e1, e2):
    e = np.concatenate([e1, e2])
    d = cosine_similarity(e, e)
    import matplotlib.pyplot as plt
    plt.imshow(d, interpolation='none')
