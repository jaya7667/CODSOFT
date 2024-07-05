#Using CountVectorizer with Naive Bayes Classifier
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('/workspaces/CODSOFT/Task 1/spam.csv',encoding='latin-1')
df.head()
df.describe()

spam_messages = df[df["v1"]=="spam"]
spam_messages.head()
spam_messages.describe()

#Splitting the data into training and testing
data_train, data_test, labels_train, labels_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0) 
print("data_train, labels_train : ",data_train.shape, labels_train.shape)
print("data_test, labels_test: ",data_test.shape, labels_test.shape)
vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(data_train)
data_test_count  = vectorizer.transform(data_test)

clf = MultinomialNB()
clf.fit(data_train_count, labels_train)
predictions = clf.predict(data_test_count)
predictions

print ("accuracy_score : ", accuracy_score(labels_test, predictions))
print ("confusion_matrix : \n", confusion_matrix(labels_test, predictions))
print (classification_report(labels_test, predictions))

     