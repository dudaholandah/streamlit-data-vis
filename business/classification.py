import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from business.mlpmodel import MLPModel
import os
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
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
      case "USDA": 
        self.data = pd.read_excel("data/usda_condesed_categories.xlsx", sheet_name="Sheet1") 
        self.label = "Category"
      case "Gluten": 
        self.data = pd.read_excel("data/gluten_dataset.xlsx", sheet_name="dataset") 
        self.label = "Product"
      case "Vegetarian/Vegan": 
        self.data = pd.read_excel("data/veg_category_usda_dataset.xlsx", sheet_name="Sheet1") 
        self.label = "vegCategory"

  def pre_processing(self, data):
    columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    data_numeric = data[columns]

    X = data_numeric.values
    attributes_dummies = data_numeric.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(X)
    data_normalized = pd.DataFrame(xscaled, columns=attributes_dummies)
    data_normalized = data_normalized.replace(np.nan,0)

    X = data_normalized.values
    pca = PCA(n_components=8)
    pca_result = pca.fit_transform(X)
    
    return pca_result

  def predict_label(self, data):

    X = self.pre_processing(data)

    model = load_model(f"{self.DIR_PATH}{self.name}_mlp.h5")
    prob = model.predict(X)
    predictions = np.argmax(prob, axis=1)

    match self.name:
      case "Vegan":
        mapping = {0:'DAIRY', 1:'EGG', 2:'FISH', 3:'MEAT', 4:'PORK', 5:'POULTRY'}
      case "USDA": 
        mapping = {0:'Fruit', 1:'Grain', 2:'Milk and Dairy', 3:'Protein', 4:'Snacks and Desserts', 5:'Vegetable'}
      case "Gluten": 
        mapping = {0:'Gluten Containing', 1:'Gluten Free'}
      case "Vegetarian-Vegan-Omni": 
        mapping = {0:'OMNI', 1:'VEGAN', 2:'VEGETARIAN'}
    
    labeled_predictions = [mapping[key] for key in predictions]
    data['Predicted Label'] = labeled_predictions       

    return 'Predicted Label'
