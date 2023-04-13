# This file provides tools and functions
# to extract and work with the amenities field of the
# AirBnB Dataset

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from scipy.cluster.hierarchy import ward, fcluster, linkage
import gensim.downloader
from sentence_transformers import SentenceTransformer, models
from utils.textProcessing import tokenize, uncontract, lowercase
from utils.tokenProcessing import remove_numbers, remove_puctuation_only, remove_stop_words, lemmatize
from sklearn.metrics.pairwise import cosine_distances



class AmenitiesClustering():
    """
    This class provide the system to execute the extraction of the amenities,
    by generating the clusters and then generating a vector for each element,
    to tell if they have an amenity in that cluster.

    Given the amenities structure for 1 listing:
    amenities: ['Big Tv', 'mega hairdryer', 'JBL audio system', ... ]
    we get the following processing:
    'Big Tv' =token_normalization> ['big', 'tv'] =w2v> [embedding1, embedding2] =mean> embedding

    It works by:
    1. extracting the unique amenities encoding using a w2v model
    2. it the create an average representation for the cluster embedding by averaging the representations
    3. for each listing check for each cluster if it has an amenity near the cluster representation
    4. you get in output a list of array containing if the listing have an amenity in such cluster
    """
    mProcessing = [
            remove_puctuation_only, 
            remove_numbers, 
            remove_stop_words, 
            lemmatize
    ]

    def __init__(
            self, 
            modelName='glove-wiki-gigaword',  # Gensim model name
            embeddingSize=300,                # size of the mode embedding output
            clusteringThreashold=0.5,         # Clustering distance Threashold
            clusteringTokensPercentual=1.0,   # Percentual of token to use to produce the clustering
            presenceThreashold=0.2,           # The distance of which the appartence of an amenity to a cluster is extrapolated
        ) -> None:
        self.mPresenceThreashold = presenceThreashold
        self.mClusteringThreashold = clusteringThreashold
        self.mClusteringTokensPercentual = clusteringTokensPercentual
        self.mModelName = modelName
        self.mEmbeddingSize = embeddingSize
        self.model = gensim.downloader.load(f'{modelName}-{embeddingSize}')

    def fit(self, df):
        tokens = self.to_tokens(df)
        tokens_flat = list(itertools.chain.from_iterable(tokens))
        unique_tokens = self.uniques(tokens_flat)
        vectorized_uniques_amenities = self.vectorize(unique_tokens)
        self.mClusters = self.get_clusters_dict(vectorized_uniques_amenities)

    def transform(self, df):
        tokens = self.to_tokens(df)
        return self.extract_cluster_presence(tokens, self.mClusters)
    
    def fit_transform(self, df):
        tokens = self.to_tokens(df)
        tokens_flat = list(itertools.chain.from_iterable(tokens))
        unique_tokens = self.uniques(tokens_flat)
        vectorized_uniques_amenities = self.vectorize(unique_tokens)
        self.mClusters = self.get_clusters_dict(vectorized_uniques_amenities)
        return self.extract_cluster_presence(tokens, self.mClusters)
    
    def extract_cluster_presence(self, tokenized_amenities):
        return [self.extract_cluster_presence_single(a) for a in tokenized_amenities]

    def extract_cluster_presence_single(self, tokenized_amenities):
        cluster_presence = np.zeros(len(self.mClusters))
        for amenities_list in tokenized_amenities:
            encoded_amenities = self.w2v.most_similar(positive=amenities_list, topn=1)[0]
            for idx, cluster_vector in self.mClusters:
                distance = cosine_distances(encoded_amenities, cluster_vector)
                cluster_presence[idx] = 1 if distance < self.mPresenceThreashold else 0
        return cluster_presence
        
    def get_clusters_dict(self, vectorized_tokens, clusters):
        cluster_dict = {}
        cluster_words = {}
        for c in clusters:
            if c not in cluster_dict: cluster_dict[c] = []
            cluster_dict[c] += [vectorized_tokens[c]]

            if c not in cluster_words: cluster_words[c] = []
            cluster_words[c] +=[]

        for idx,  c in cluster_dict:
            cluster_dict[idx] = np.stack(c).mean(axis=1)

        return cluster_dict

    def extract_cluster(self, vectorized_amenities):
        Z = linkage(vectorized_amenities,'average', metric='cosine')
        clusters = fcluster(Z, threashold, criterion='distance')
        return clusters

    def tokenize(self, listings):
        tokens_list = []
        for idx, listing in tqdm(listings.iterrows()):
            amenities_list = json.loads(listing['amenities'])
            amenities = [ tokenize(lowercase(uncontract(a))) for a in  amenities_list]
            tokens_list.append(amenities)

        return tokens_list
    
    def uniques(amenities):
        res_list = []
        test = []
        for item in amenities: 
            if "".join(item) not in test:
                res_list.append(item)
                test.append("".join(item))

        return res_list
    
    def to_tokens(self, df):
        docs_tokens = self.tokenize(df)
        docs_tokens = [self.apply_process_list(doc, self.mClusteringProcessing) for doc in docs_tokens]
        return docs_tokens

    def apply_process_list(self, tokens):
        for p in self.mProcessing:
            tokens = [p(token) for token in tokens]
        return tokens

    def vectorize(self, docs):  
        documents_vector = []
        for d in docs:
            vectorized_amenity_tokens = [self.w2v_model[t] for t in d if t in self.w2v_model]
            documents_vector.append(np.zeros(shape=(self.mEmbeddingSize)) if len(vectorized_amenity_tokens) == 0 else np.mean(vectorized_amenity_tokens, axis=0))
        return documents_vector



class SentenceModel():
    def __init__(self, modelName='distilbert-base-nli-mean-tokens') -> None:
        self.model = SentenceTransformer(modelName)

    def encode(self, df):
        encodings = {}
        for idx, listing in tqdm(df.iterrows()):
            amenities = json.loads(listing['amenities'])
            a_string = " ".join(amenities)
            encoding = self.model.encode(a_string)
            encodings[idx] = encoding
        
        return encodings
