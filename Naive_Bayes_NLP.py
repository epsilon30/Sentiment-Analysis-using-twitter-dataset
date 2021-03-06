# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:29:55 2019

@author: Epsilon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('twitter_x_y_train.csv')

X=dataset.iloc[:,7].values
Y=dataset.iloc[:,1].values


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
Y=label.fit_transform(Y)


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(10981):
    x=re.sub('@[\w]*',' ',X[i])
    x=x.lower()
    x=x.split()
    ps=PorterStemmer()
    x= [ps.stem(word)for word in x if not word in set(stopwords.words('english'))]
    x=' '.join(x)
    corpus.append(x)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1000)
X=cv.fit_transform(corpus).toarray()



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test,y_pred)