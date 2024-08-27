import re
from unidecode import unidecode
import pandas as pd
from sklearn import preprocessing
import numpy as np
import streamlit as st

def pre_process_string(text):
  text = re.sub(r'[\[\].,():%-]+', " ", text)
  text = re.sub(r'[\s]+', " ", text)
  text = unidecode(text.strip().lower())
  return text

def define_type(column):
  if pd.api.types.is_numeric_dtype(column):
    return "Numeric"
  elif column.apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).mode()[0]:
    return "List of strings"
  elif column.apply(lambda x: isinstance(x, str) and ',' in x).mode()[0]:
    return "Comma-separated string"
  elif column.apply(lambda x: isinstance(x, str)).mode()[0]:
    return "Simple string"
  else:
    return "Unknown type"

def vocab_list(data):
  vocab = dict()
  for obj in data:
    for word in obj.split(","):
      word = pre_process_string(word)
      if len(word) > 1 and word not in vocab: 
        vocab[word] = len(vocab)

  vocab = sorted(vocab)
  return vocab

def pre_processing_strings(data):
  vocab = dict()
  for att in data:
    for obj in data[att]:
      for word in obj.split(","):
        word = pre_process_string(word)
        if len(word) > 1 and word not in vocab: 
          vocab[word] = len(vocab)

  vocab = sorted(vocab)
  ingredients_normalized = {}

  for word in vocab:
    ingredients_normalized[word] = []

  for att in data:
    for obj in data[att]:
      aux = obj.split(",")
      for i in range(len(aux)):
        aux[i] = pre_process_string(aux[i])
      for word in vocab:
        if word in aux: ingredients_normalized[word].append(1)
        else: ingredients_normalized[word].append(0)   

  ingredients_normalized = pd.DataFrame(ingredients_normalized)

  return ingredients_normalized

def pre_processing_numbers(data):
  attributes_dummies = data.columns
  normalize = preprocessing.MinMaxScaler()
  xscaled = normalize.fit_transform(data.values)
  data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
  data_normalized = data_normalized.replace(np.nan,0)
  return data_normalized

def selected_points_to_dataframe(selected_points):
  if selected_points == []: 
    return pd.DataFrame()
  
  data = st.session_state['selected_data']
  selected_rows = []
  selected_idx = set()

  for i in data.index:
    for pt in selected_points:
      if abs(data['x'][i] - pt['x']) < 1e-4 and abs(data['y'][i] - pt['y']) < 1e-4 and i not in selected_idx:
        selected_idx.add(i)
        selected_rows.append(pd.DataFrame([data.iloc[i]]))

  points_dataframe = pd.concat(selected_rows, ignore_index=False)
  points_dataframe = points_dataframe.drop(['x', 'y'], axis=1)
  return points_dataframe


def filter_dataframe(points_dataframe : pd.DataFrame, creating_label=False):

  if points_dataframe.empty:
    print("DataFrame is Empty!!!")
    st.session_state['df_filtered'] = pd.DataFrame()
    return

  data = st.session_state['selected_data']
  df = st.session_state['df']

  for i in points_dataframe.index:
    if creating_label:
      data.loc[i, st.session_state['label']] = st.session_state['new_label']
      df.loc[i, st.session_state['label']] = st.session_state['new_label']

  st.session_state['df_filtered'] = points_dataframe.copy()
  st.session_state['local_vis_ready'] = True
