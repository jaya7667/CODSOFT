#Using Tfdf vectorizer with Naive Bayes Classifier
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sklearn.metrics as m
from sklearn.preprocessing import LabelEncoder
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dataset=pd.read_csv('/workspaces/CODSOFT/Task 1/spam.csv',encoding='latin-1')
dataset

sent=dataset.iloc[:,[1]]['v2']
sent

label=dataset.iloc[:,[0]]['v1']
label

le=LabelEncoder()
label=le.fit_transform(label)
label

len(set(stopwords.words('english')))
stem=PorterStemmer()
sent

sentences=[]
for sen in sent:
  senti=re.sub('[^A-Za-z]',' ',sen)
  senti=senti.lower()
  words=word_tokenize(senti)
  word=[stem.stem(i) for i in words if i not in stopwords.words('english')]
  senti=' '.join(word)
  sentences.append(senti)
sentences

tfidf=TfidfVectorizer(max_features=5000)
features=tfidf.fit_transform(sentences)
features=features.toarray()
features

#Splitting the data into training and testing
model=MultinomialNB()
feature_train,feature_test,label_train,label_test=train_test_split(features,label,test_size=0.2,random_state=7)
model.fit(feature_train,label_train)

label_pred=model.predict(feature_test)
label_pred
label_test
m.accuracy_score(label_test,label_pred)


print(m.classification_report(label_test,label_pred))