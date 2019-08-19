# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:16:32 2019

@author: Ditskih
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
df = pd.read_csv("D:\\Private Property\\Data Kuliah\\Akademis\\Webminings\\Webmining\\Data\\Berita\\2014-03.csv")
df = df.drop(columns="[M] category: ")
df = df.drop(columns="[M] comments: ")
df = df.drop(columns="[M] creator: ")
df = df.drop(columns="[M] encoded: ")
df = df.drop(columns="[M] link: ")
df = df.drop(columns="[M] pubdate: ")
token= word_tokenize(df)


#df = df.drop(columns="Title")
df.to_csv("dicobaduluajaya.csv", index=False, encoding='utf8')
# Output data to an Excel file.
#print(dataframe1['Title'])
#print(dataframe1['Text'])