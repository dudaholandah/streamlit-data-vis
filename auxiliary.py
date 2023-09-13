from sklearn import preprocessing
import pandas as pd
import numpy as np
from unidecode import unidecode
import re

def pre_processing_data(df, label, attributes):
  data, data_label = separating_data(df, label, attributes)
  data = normalizing_data(data)
  return final_data(data, data_label) # X, y, data

def pre_process(text):
  text = re.sub(r'[.,():%-]+', " ", text)
  text = re.sub(r'[\s]+', " ", text)
  text = unidecode(text.strip().lower())
  return text

def separating_data(df, label, attributes):

  for index, row in df.iterrows():
    processed_label = df[label][index].strip()
    df.at[index,label] = processed_label

  data = df[attributes].copy()
  if label == "": data_label = pd.DataFrame()
  else: data_label = df[label].copy()
  return data, data_label

def tfidf(X):
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer()

  corpus = []

  for col in X:
    for text in X[col]:
      corpus.append(text)
  
  transformFit = vectorizer.fit_transform(corpus)

  df = pd.DataFrame(transformFit.toarray().transpose(), vectorizer.get_feature_names_out()).T
  return df

def one_hot_encoding(X):
  vocab = dict()
  for att in X:
    for obj in X[att]:
      for word in obj.split(","):
        word = pre_process(word)
        if len(word) > 1 and word not in vocab: 
          vocab[word] = len(vocab)

  vocab = sorted(vocab)
  
  ingredients_normalized = {}

  for word in sorted(vocab):
    ingredients_normalized[word] = []

  for att in X:
    for obj in X[att]:
      aux = obj.split(",")
      for i in range(len(aux)):
        aux[i] = pre_process(aux[i])
      for word in sorted(vocab):
        if word in aux: ingredients_normalized[word].append(1)
        else: ingredients_normalized[word].append(0)   

  ingredients_normalized = pd.DataFrame(ingredients_normalized)

  return ingredients_normalized

def normalizing_data(df):

  X_num = []
  X_obj = []
  
  for att in df:
    if df[att].dtype == "float64": X_num.append(att)
    else: X_obj.append(att)     
  
  X_num = df[df.columns.intersection(X_num)]
  X_obj = df[df.columns.intersection(X_obj)]

  attributes_dummies = X_num.columns
  normalize = preprocessing.MinMaxScaler()
  xscaled = normalize.fit_transform(X_num.values)
  data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
  data_normalized = data_normalized.replace(np.nan,0)
  
  if not X_obj.empty: 
    '''pode utilizar one hot encoding ou tfidf'''
    # obj_normalized = one_hot_encoding(X_obj) 
    obj_normalized = tfidf(X_obj)
    data_normalized = pd.concat([data_normalized,obj_normalized], axis=1)
    data_normalized = data_normalized.replace(np.nan,0)

  return data_normalized

def final_data(data, data_label):
  raw_data = data
  raw_data = raw_data.replace(np.nan,0)

  for i in range(data_label.shape[0]):
    data_label[i] = data_label[i].rstrip()

  X = raw_data.values
  y = data_label.values
  all_data = raw_data.join(data_label)

  return X, y, all_data

def get_x_y(data):
  data[1:4, :]
  X = data[:,0]
  y = data[:,1]
  return X.tolist(), y.tolist()
