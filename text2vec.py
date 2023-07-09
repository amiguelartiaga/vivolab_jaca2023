

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


from sentence_transformers import SentenceTransformer
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("hiiamsid/sentence_similarity_spanish_es")
model = AutoModel.from_pretrained("hiiamsid/sentence_similarity_spanish_es")

def get_tokenizer():
    return tokenizer

def text_to_vector(text):
    # Tokenize sentences
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    #   print(sentence_embeddings)
    x = sentence_embeddings.cpu().numpy()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    return xn


def text_to_vector_by_phrases(texto, n=4, m=2):
    text_ = texto.split('.')
    if len(text_) < 2*n:
        return text_to_vector(texto)
    else:
        v_ = []
        for i in range(0, len(text_)-3, m):
            texto = '. '.join(text_[i:i+3]).strip()
            #print(texto)
            v_.append(text_to_vector(texto))
        return np.concatenate(v_)
    
        
import pandas as pd
from tqdm import tqdm
def pandas_text_vector(datos, column_text='text', column_vector='text-vector'):
    v = pd.DataFrame(columns=[column_vector])
    v[column_vector] = v[column_vector].astype(object)
    for i in tqdm(datos.index):
        texto = datos[column_text][i]        
        v.loc[i,column_vector] = text_to_vector(texto)    
    datos[column_vector] = v[column_vector]
    return datos

import pandas as pd
from tqdm import tqdm
def pandas_text_vector_by_phrases(datos, column_text='text', column_vector='text-vector', n=3):
    v = pd.DataFrame(columns=[column_vector])
    v[column_vector] = v[column_vector].astype(object)

    for i in tqdm(datos.index):
        texto = datos[column_text][i]        
        v.loc[i,column_vector] = text_to_vector_by_phrases(texto)    
    datos[column_vector] = v[column_vector]
    return datos


def pandas_text_similarity(datos, column, search, searchname=None):
    if searchname is None:
        searchname = column+' tsim '+search    
    search = text_to_vector(search).T
    for i in tqdm(datos.index):
        vector = datos[column][i]  
        if not np.isnan(vector).any():
            s = np.dot(vector, search).max()            
            datos.at[i, searchname] = s
    return datos


