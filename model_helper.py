from __future__ import division,print_function,absolute_import
import os


import numpy as np
import keras
import keras.models
from keras.models import model_from_json
import tensorflow as tf

from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

import pickle
import pandas as pd
# import Stemmer
# stemmer = Stemmer.Stemmer('english')
import re 
import ult
np.random.seed = 9

def read_json_file(path, mode):
    f = open(path,mode)
    json_ = f.read()
    f.close()
    return json_

def init(config): 
    model_json = read_json_file(config['MODEL_JSON_DIR'],'r')
    saved_model = model_from_json(model_json)
    saved_model.load_weights(config['MODEL_WEIGHTS_DIR'])
    saved_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    tf_graph = tf.get_default_graph()

    return saved_model,tf_graph

class TextTransformer:

    def __init__(self,config):
       self.contractions = ult.get_contractions()
       self.config = config
    
    #Cleaning 
    
    #print(contractions)
    def expandShort(self,sent):
        for word in sent.split():
            if word.lower() in self.contractions:
                sent = sent.replace(word, self.contractions[word.lower()])
        return sent

    def cleanText(self,sent):
        sent = sent.replace("\\n"," ")            
        sent = sent.replace("\\xa0"," ") #magic space lol
        sent = sent.replace("\\xc2"," ") #space
        sent = re.sub(r"(@[A-Za-z]+)|([\t])", "",sent)
        #sent = expandShort(sent.strip().lower())
        sent = re.sub(r'[^\w]', ' ', sent)
        sent = re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])", " ", sent)
        sent = self.expandShort(sent.strip().lower())
        ws = [w for w in sent.strip().split(' ') if w is not ''] # remove double space
        return " ".join(ws)

    #stemming
    def stem(self,s):
        ws = s.split(' ')
        #ws = stemmer.stemWords(ws)
        return " ".join(ws)

    def removebackSlash(self,sent):
        return sent.replace('\\\\','\\').replace("\\"," ")

    def pre_proc_text_prod(self,text):
        c = self.cleanText(text)
        c = self.removebackSlash(c)
        c = self.stem(c)
        data = self.tokanized_prod([c])
        return data

    def tokanized_prod(self,texts):
        saved_tokenizer = pickle.load(open(self.config['TOKENZER_PATH'], "rb"))
        sequences = saved_tokenizer.texts_to_sequences(texts)
        #padded_texts = pad_sequences(sequences, maxlen=config['MAX_SEQUENCE_LENGTH'])
        data = saved_tokenizer.sequences_to_matrix(sequences,mode='tfidf')
        return data