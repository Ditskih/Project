# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:38:52 2019

@author: Ditskih
"""
import os
import json
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection as GRP
import numpy as np
import sys
sys.path.insert(0, 'FCMeans')
from fcmeans import fcmeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pandas as pd

def my_preprocessor(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

def my_tokenizer(tweet):
    words = word_tokenize(tweet)
    tokens=[]
    for w in words:
        #replace two or more with two occurrences
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        w = pattern.sub(r"\1\1", w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #choose words with a pattern
        val = re.search(r"^[a-zA-Z0-9][a-zA-Z0-9]*$", w)
        #add tokens
        if(w in ['AT_USER','URL'] or val is None):
            continue
        else:
            tokens.append(w.lower())

    return tokens


for i in range (1):

    # -------
    # Loading
    # -------
    print ("Loading dataset .... ")
    df = csv.reader(open("D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\program\\Program1\\Program\\nyoba\\dicobaduluajafix.csv", encoding="utf8"))
    data = []
    for column in df:
        data.append(column[0].strip() + ' ' + column[1].strip())

    # -----------
    # Vectorizing : Preprocessing, Tokenizing, Filtering, Weighting
    # -----------
    print ("Vectorizing .....")

    data_file = csv.reader(open('D:\Private Property\Data Kuliah\Akademis\Skripsweet\program\Program1\Program\\nyoba\\stopwords_id.csv'))
    stopwords = []
    for column in data_file:
        stopwords.append(column[0])
    my_stop_words = stopwords + ['untuk','toko','nya','false','none''0', '01', '02', '0223', '03', '04', '05', '06', '07', '08', '09',
                                 '0pertandingan', '1', '10', '100', '1001', '101', '102', '1020', '103', '104', '105', '106', '108', '109',
                                 '10th', '11', '110', '112', '113', '115', '12', '120', '121', '122', '123', '125', '129', '13', '130', '131',
                                 '132', '135', '136', '137', '138', '139', '14', '140', '141', '142', '145', '148', '15', '150', '1500',
                                 '152', '153', '154', '155', '157', '16', '160', '161', '162', '165', '166', '168', '17', '170', '1700',
                                 '172', '1731', '175', '1763', '18', '180', '1800', '181', '184', '1848', '185', '187', '19', '190',
                                 '1906', '191', '1930', '1936', '1945', '1947', '1948', '1949', '1950', '1954', '1955', '1958', '196',
                                 '1961', '1962', '1964', '1965', '1967', '1968', '1972', '1973', '1974', '1984', '1985', '1987', '199',
                                 '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1a', '1musim', '1st', '2', '20',
                                 '200', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '200cc', '201', '2010',
                                 '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2025',
                                 '2041', '2045', '205', '2050', '207', '21', '210', '211', '215', '22', '221', '223', '225', '227', '229',
                                 '23', '230', '234', '235', '238', '239', '24', '240', '241', '25', '250', '250cc', '2560x1440', '258', '259',
                                 '26', '260', '263', '265', '267', '268', '27', '278', '28', '280', '282', '283', '284', '286', '29',
                                 '2pm', '3', '30', '300', '306', '308', '31', '310', '315', '32', '33', '330', '34', '345', '35', '350',
                                 '359', '36', '360', '369', '37', '370', '378', '38', '386', '387', '39', '399', '3c', '3d', '3s', '4',
                                 '40', '400', '407', '41', '410', '42', '43', '44', '45', '450', '46', '4640', '47', '4720', '48', '480',
                                 '49', '4g', '4minute', '4x2', '4x4', '5', '50', '500', '500c', '508', '50mp', '51', '52', '53', '54', '55',
                                 '550', '56', '560', '57', '58', '59', '595', '5c', '5g', '5s', '5th', '6', '60', '600', '61', '62', '623',
                                 '625', '63', '634', '64', '640', '65', '650', '656', '66', '67', '68', '69', '69053', '6a', '6x6', '7', '70',
                                 '700', '71', '72', '720', '73', '737', '74', '7442', '75', '750', '7569', '76', '77', '78', '79', '8', '80',
                                 '800', '80an', '81', '814', '816', '82', '83', '84', '85', '8500', '86', '865', '86th', '87', '88', '889',
                                 '89', '8gb', '9', '90', '900', '91', '911', '92', '93', '94', '95', '96', '97', '98', '99', 'a', 'a3', 'a320', 'a66s', 'aa']

    vectorizer = TfidfVectorizer(preprocessor=my_preprocessor,tokenizer=my_tokenizer,
                                 stop_words=my_stop_words,min_df=2,max_df=0.95)
    data = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    
    #print (feature_names)
    #break
    #print (data)

    # ------------------------------------------
    # Model to Transform Data into a Lower Space
    # ------------------------------------------
    grps = GRP(n_components = 5)
    new_data = grps.fit_transform(data)

    # Learning
    # --------
    for n_topics in range(100,110,10):
        print ("Learning ...." + str(n_topics))
        
        #membership (u) calculation in the lower space
        m=1.5
        cntr, u= fcmeans(new_data.T, n_topics, m, error=0.005, maxiter=1000)

        #centroid (cntr) calculation in the original space
        temp = csr_matrix(np.ones((data.shape[1],1)).dot(np.atleast_2d(u.sum(axis=1))).T)
        u = csr_matrix(u)
        cntr = np.asarray(u.dot(data) / temp)
        
        ''' 
        # Find centroids for initialization
        svd = TruncatedSVD(n_components = n_topics)
        svd.fit(new_data)
        cntr = svd.components_
        #cntr[cntr<0.001]=0.0
        
        # Find centroids by FCM
        cntr, u = fcmeans(new_data.T, n_topics, m=1.5, error=0.005, maxiter=1000, init=cntr.T)
        cntr = np.asarray(cntr)
        ''' 
        # Prints topics
        n_top_words = 10
        hasil = open('D:\\Private Property\\Data Kuliah\\Akademis\\Skripsweet\\program\\Program1\\Program\\nyoba\\topikgrp' + str(n_topics) + ".txt", 'w')
        for topic_idx, topic in enumerate(cntr):
            print("Topic " + str(topic_idx) + " : " + " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            hasil.write(""+" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) + "\n")
        hasil.close()
