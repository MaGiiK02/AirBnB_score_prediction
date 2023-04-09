# This file provides tools and functions
# to extract and work with the amenities field of the
# AirBnB Dataset

import json
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


   
def extract_amenities(string_amentities):
    return json.loads(string_amentities)

class AmenitiesProcessor():
    def __init__(self, method, method_settings) -> None:
        self.mMethod = method
        self.mMethodSettings = method_settings
        self.w2vModel = None

        if self.mMethod == 'bert':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        elif self.mMethod == 'oneshot': pass
        elif self.mMethod == 'w2v': pass
        else:raise Exception("AmenitiesProcessor.encode: Invalid method proposed")

    def encode_one(self, amenities_arrays):
        if self.mMethod == 'bert': pass
        elif self.mMethod == 'w2v': pass
        else:raise Exception("AmenitiesProcessor.encode: Invalid method proposed") 
  
    def encode_all(self, dataframe):
        amenities = dataframe[['id', 'amenities']]
        amenities['amenities'] = amenities['amenities'].apply(extract_amenities)
        encoded_amenities = []

        for idx, row in tqdm(amenities.sample(3).iterrows()):
            amenities_values = row['amenities']
            tester = self.encode_one(amenities_values)
            encoded_amenities.append(tester)

        return encoded_amenities

def encode_amenities_w2v(self, amenities):
    if self.w2vModel == None: self.w2vModel = gensim.downloader.load('word2vec-google-news-300')

    encoded_amenities = []
    for t in amenities:
        # if token in the model
        if t in self.w2vModel: encoded_amenities.append(self.w2vModel[t])

    return np.zeros(shape=(300)) if len(encoded_amenities) == 0 else np.mean(encoded_amenities, axis=0)

# We create a single string of amenities that we compute with bert
def encode_amentities_bert(self, amenities):
    amenities_string = ' '.join(amenities.sort()).lower()
    encoded_input = self.tokenizer(amenities_string, return_tensors='pt')
    return self.model(**encoded_input).last_hidden_state