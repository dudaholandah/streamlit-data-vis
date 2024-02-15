import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from business.mlpmodel import MLPModel
import os
from tensorflow.keras.models import load_model
import re

class Classification:

  SEP = os.path.sep + 'data\\'
  DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + SEP

  def __init__(self, name):
    self.name = name
    match name:
      case "Vegan": 
        self.data = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos") 
        self.label = "Classification"
      # case "USDA": 
      #   self.data = pd.read_excel("data/usda_dataset.xlsx", sheet_name="usda_dataset") 
      #   self.label = "Category"
      case "Gluten": 
        self.data = pd.read_excel("data/gluten_dataset.xlsx", sheet_name="dataset") 
        self.label = "Product"
      case "Vegetarian/Vegan": 
        self.data = pd.read_excel("data/veg_category_usda_dataset.xlsx", sheet_name="Sheet1") 
        self.label = "vegCategory"

  def normalize_data(self, data):
    columns_to_search = ['Carbohydrate', 'Fiber', 'Protein', 'Sugar', 'Total Fat', 'Calcium', 'Sodium', 'Zinc']
    pattern = '|'.join(columns_to_search)
    pattern = re.compile(pattern, flags=re.IGNORECASE)

    matched_columns = data.filter(regex=pattern)
    data_filtred = matched_columns.reindex(sorted(matched_columns.columns), axis=1)

    X = data_filtred.values
    attributes_dummies = data_filtred.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(X)
    data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)

    data_normalized = data_normalized.replace(np.nan,0)
    return data_normalized

  def get_X(self, data):
    embedding = data.copy()
    embedding = embedding.replace(np.nan,0)
    return embedding.values

  def predict_label(self, data):

    data_normalized = self.normalize_data(data)
    X = self.get_X(data_normalized)
    
    model = load_model(f"{self.DIR_PATH}{self.name}_mlp.h5")
    prob = model.predict(X)
    predictions = np.argmax(prob, axis=1)

    match self.name:
      case "Vegan":
        mapping = {0:'DAIRY', 1:'EGG', 2:'FISH', 3:'MEAT', 4:'PORK', 5:'POULTRY'}
      # case "USDA": 
      #   mapping = {0:'OMNI', 1:'VEGAN', 2:'VEGETARIAN'}
      case "Gluten": 
        mapping = {0:'Gluten Containing', 1:'Gluten Free'}
      case "Vegetarian-Vegan-Omni": 
        mapping = {0:'OMNI', 1:'VEGAN', 2:'VEGETARIAN'}
    
    labeled_predictions = [mapping[key] for key in predictions]
    data['Predicted Label'] = labeled_predictions       

    return 'Predicted Label'
