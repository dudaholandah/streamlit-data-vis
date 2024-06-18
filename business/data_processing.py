import re
from unidecode import unidecode
import pandas as pd
from sklearn import preprocessing
import numpy as np

class DataProcessing:

  def __init__(self):
    pass

  def pre_process(self, text):
    text = re.sub(r'[\[\].,():%-]+', " ", text)
    text = re.sub(r'[\s]+', " ", text)
    text = unidecode(text.strip().lower())
    return text
  
  def define_type(self, column):
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
    
  def pre_processing_strings(self, data):
    vocab = dict()
    for att in data:
      for obj in data[att]:
        for word in obj.split(","):
          word = self.pre_process(word)
          if len(word) > 1 and word not in vocab: 
            vocab[word] = len(vocab)

    vocab = sorted(vocab)
    
    ingredients_normalized = {}

    for word in sorted(vocab):
      ingredients_normalized[word] = []

    for att in data:
      for obj in data[att]:
        aux = obj.split(",")
        for i in range(len(aux)):
          aux[i] = self.pre_process(aux[i])
        for word in sorted(vocab):
          if word in aux: ingredients_normalized[word].append(1)
          else: ingredients_normalized[word].append(0)   

    ingredients_normalized = pd.DataFrame(ingredients_normalized)

    return ingredients_normalized
  
  def pre_processing_numbers(self, data):
    attributes_dummies = data.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(data.values)
    data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
    data_normalized = data_normalized.replace(np.nan,0)
    return data_normalized