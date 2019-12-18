# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:29:24 2019

@author: khoironi
"""
#source from yudiwbs https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/

#%%
#load data

from tkinter import *
import matplotlib.pyplot as plt
from tkinter import filedialog
from collections import Counter
import csv
import re
import string
frequency = {}

namaFile = "dataset.csv"
data = []
label = []
with open(namaFile, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader) #skip header
    for row in reader:
        data.append(row[0])
        label.append(row[1])
 
print("jumlah data:{}".format(len(data)))
print(Counter(label))

#%%
#random urutan dan split ke data training dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( data, label, test_size=0.2, random_state=123)
 
print("Data training:")
print(len(X_train))
print(Counter(y_train))
 
print("Data testing:")
print(len(X_test))
print(Counter(y_test))
 
#%%
#transform ke tfidf dan train dengan naive bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
text_clf.fit(X_train, y_train)
#%%
# coba prediksi data baru

namaFilee = "datates.csv"
userid = []
tweet = []
csvfile = [i.split() for i in namaFilee]

with open(namaFilee, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader) #skip header
    for row in reader:
        userid.append(row[0])
        tweet.append(row[1])

        
pred = text_clf.predict(tweet)
print("Hasil prediksi {}".format(pred))
print("number tweet {}".format(userid))
#%%
#hitung akurasi data test
import numpy as np
pred = text_clf.predict(X_test)
akurasi = np.mean(pred==y_test)
print("Akurasi: {}".format(akurasi))




#plt.plot(X_test, y_test)
#plt.show()

 
#kemunculan = nltk.FreqDist(label)
#kemunculan.plot(30,cumulative=False)
#plt.show()