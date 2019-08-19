# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:42:57 2019

@author: Ditskih
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, 'FCMeans')
#from fcmeans import fcmeans
from fcmeans import fcmeans

# GET the data
## Memuat Data Training
dataset = pd.read_csv("D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\Konferensi\\Program konferensi\\data\\beritafebruari.csv", usecols=["Title", "Text"])

# EXPLORE the data
## Menampilkan lima data pertama
dataset.tail()

## Pembersihan
stopwords = pd.read_csv("D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\Konferensi\\Program konferensi\\stopwords_id.csv")
stopwords = np.append(stopwords, "rt")

def clean_text(tweet):
    
    # Convert to lower case
    tweet = tweet.strip().lower()
    
    # Clean www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    # Clean @username
    tweet = re.sub('@[^\s]+','',tweet)
    # Clean # from #word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Replace three or more into two alphabet occurrences
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    tweet = pattern.sub(r"\1\1", tweet)
    # Strip digit from word
    tweet = re.sub("[0-9]","", tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Remove stop and short words
    tokenizer = re.compile( r"(?u)\b\w\w+\b" )
    tokens = []
    for tok in tokenizer.findall( tweet ):
        if (tok in stopwords or len(tok) < 2):
            continue
        else:
            tokens.append(tok)

    tweet = " ".join(tokens)
    return tweet.strip()

dataset['Tweets'] = dataset['Title'] + ' ' + dataset['Text']
dataset['Tweets'] = dataset['Tweets'].astype(str)
dataset['Tweets'] = dataset['Tweets'].map(lambda x: clean_text(x))
dataset = dataset[dataset['Tweets'].apply(lambda x: len(x.split()) >=3)]
dataset.tail()
 

# MODEL the data
## Pra Pengolahan - Tokenisasi, Pembobotan, Vektorisasi
vectorizer = TfidfVectorizer(min_df=2,max_df=0.95)
data = vectorizer.fit_transform(dataset['Tweets'])
feature_names = vectorizer.get_feature_names()
#print(feature_names)
#print(data.shape)

## Reduksi Dimensi
svd = TruncatedSVD(n_components = 5)


## Pemilihan Model
def calculate_coherence(w2v_model, term_rankings):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            if pair[0] in w2v_model.wv.vocab:
                if pair[1] in w2v_model.wv.vocab:
                    pair_scores.append( w2v_model.wv.similarity(pair[0], pair[1]) )
                else:
                    pair_scores.append(0)
            else:
                pair_scores.append(0)
                
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

### Memuat Model Word2Vec
w2v_model = gensim.models.Word2Vec.load("D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\Konferensi\\Program konferensi\\modelling\\w2v-model.bin")

### Menentukan Nilai Coherence
n_top_words = 10
num_topics = []
coherences = []
coherences_mean = []
for n_topics in range(10,20,10):
    print("The number of topics is " + str(n_topics))
    
    coherence_sim = []
    for i in range(1,6):
        print("The simulation " + str(i))
        
        #membership (u) calculation in the lower space
        m=1.5
        cntr, u = fcmeans(svd.fit_transform(data).T, n_topics, m, error=0.005, maxiter=1000, init=None)
        
        #centroid (cntr) calculation in the original space
        u = u ** m
        temp = csr_matrix(np.ones((data.shape[1],1)).dot(np.atleast_2d(u.sum(axis=1))).T)
        u = csr_matrix(u)
        cntr = np.asarray(u.dot(data) / temp)
        
        #coherence calculation
        coherence_topic = []
        for topic_idx, topic in enumerate(cntr):
            top_terms = []
            top_terms.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            coherence_topic.append(calculate_coherence(w2v_model, top_terms))
            #print(top_terms)
        
        coherence_sim.append(sum(coherence_topic) / len(coherence_topic))
    
    num_topics.append(n_topics)
    coherences.append(coherence_sim)
    coherences_mean.append(sum(coherence_sim) / len(coherence_sim))
    
### Menentukan Jumlah Topik Terbaik
cmax = max(coherences_mean)
best_num_topics = num_topics[coherences_mean.index(cmax)]
print("The number of topics is %d with coherence of %f" % (best_num_topics, cmax))

## Pemilihan Model - Visualisasi
df = pd.DataFrame.from_records(coherences).T
df.columns = num_topics
boxplot = df.boxplot(grid=False)
boxplot.set_xlabel("Number of Topics")
boxplot.set_ylabel("Coherence")

## Penyimpanan Coherence ke File Excel
df.to_excel("D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\Konferensi\\Program konferensi\\hasil maret\\fcm\\beritaEFCM.xlsx")