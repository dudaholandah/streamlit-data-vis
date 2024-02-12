from sklearn import preprocessing
import pandas as pd
import numpy as np
from unidecode import unidecode
import re
import os
import pickle

def pre_processing_data(df, label, attributes, legend):
  data, data_label, data_legend = separating_data(df, label, attributes, legend)
  data = normalizing_data(data)
  return final_data(data, data_label, data_legend) # X, y, data

def predict_label(data, X, choice):

  PATH = os.getcwd() + "/classification/data/"

  if choice == "(1)Vegan/Vegetarian/Omni":
    saved_rf_model = pickle.load(open(PATH + "veg_category_rf.pkl", 'rb'))
    predictions = saved_rf_model.predict(X)

    mapping = {0:'OMNI', 1:'VEGAN', 2:'VEGETARIAN'}
    labeled_predictions = [mapping[key] for key in predictions]

    data['predicted_label'] = labeled_predictions
    label = 'predicted_label'


  return data, label

  # elif choice == "(2)Gluten Containing/Gluten Free":
  #   pass
  # else: 
  #   pass


# def pre_processing_data_from_classification(df, label, attributes, legend):
#   if label == "(1)Vegan/Vegetarian/Omni":
#     # veg_category dataset
#     df_class = pd.read_excel(f"{os.getcwd()}/data/veg_category_usda_dataset.xlsx", sheet_name='Sheet1')
#     data_label = df['vegCategory'].copy()
    
#   elif label == "(2)Gluten Containing/Gluten Free":
#     # gluten dataset
#     df_class = pd.read_excel(f"{os.getcwd()}/data/gluten_dataset.xlsx", sheet_name='dataset')
#     data_label = df['Product']
#   else: 
#     # vegan dataset
#     df_class = pd.read_excel(f"{os.getcwd()}/data/vegan_dataset.xlsx", sheet_name='Veganos')
#     data_label = pd.DataFrame(df['Classification'])

#   print("oi1")

#   data = df[attributes].copy()

#   print("oi2")
#   data_legend = df[legend].copy()
#   print("oi3")
#   data = normalizing_data(data)
#   print("oi4")
#   return final_data(data, data_label, data_legend) # X, y, data

def pre_process(text):
  text = re.sub(r'[.,():%-]+', " ", text)
  text = re.sub(r'[\s]+', " ", text)
  text = unidecode(text.strip().lower())
  return text

def separating_data(df, label, attributes, legend):

  # # check if type of label is string
  # if not isinstance(df[label][0], str):
  #   df[label] = df[label].values.astype(str)
  # else:
  #   for index, row in df.iterrows():    
  #     processed_label = df[label][index].strip()
  #     df.at[index,label] = processed_label

  data = df[attributes].copy()

  if label == "": data_label = pd.DataFrame()
  else: data_label = df[label].copy()

  if legend == []: data_legend = pd.DataFrame()
  else:  data_legend =  df[legend].copy()
  
  return data, data_label, data_legend

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
    # isinstance(df[att], (int, float, complex))
    # df[att].dtype == "float64"
    if df[att].dtype == "float" or df[att].dtype == "int": X_num.append(att)
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

def final_data(data, data_label, data_legend):
  raw_data = data
  raw_data = raw_data.replace(np.nan,0)

  for i in range(data_label.shape[0]):
    data_label[i] = data_label[i].rstrip()

  X = raw_data.values
  y = data_label.values
  all_data = raw_data.join(data_label)

  all_data = all_data.join(data_legend)

  return X, y, all_data

def get_x_y(data):
  data[1:4, :]
  X = data[:,0]
  y = data[:,1]
  return X.tolist(), y.tolist()
