import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlpmodel import MLPModel
import os
import re

class Training:

  SEP = os.path.sep + 'data\\'
  DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + SEP

  def __init__(self, name):
    self.name = name
    match name:
      case "Vegan": 
        self.data = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos") 
        self.label = "Classification"
        for i, row_label in enumerate(self.data[self.label]):
          self.data.at[i,self.label] = re.sub(r'[0-9]+', "", row_label).strip()
      case "USDA": 
        self.data = pd.read_excel("data/usda_dataset.xlsx", sheet_name="usda_dataset") 
        self.label = "Category"
      case "Gluten": 
        self.data = pd.read_excel("data/gluten_dataset.xlsx", sheet_name="dataset") 
        self.label = "Product"
      case "Vegetarian-Vegan-Omni": 
        self.data = pd.read_excel("data/veg_category_usda_dataset.xlsx", sheet_name="Sheet1") 
        self.label = "vegCategory"

    self.normalize_nutrients()
    self.create_embedding()

    mlp_model = MLPModel(self.X, self.y, self.name + "_mlp", self.DIR_PATH, self.label_encoder) 
    mlp_model.hyperparameter_tuning()

    self.f1_macro, self.f1_micro = mlp_model.evaluate_model()

  def normalize_nutrients(self):
    columns_to_search = ['Carbohydrate', 'Fiber', 'Protein', 'Sugar', 'Total Fat', 'Calcium', 'Sodium', 'Zinc']
    pattern = '|'.join(columns_to_search)
    pattern = re.compile(pattern, flags=re.IGNORECASE)

    matched_columns = self.data.filter(regex=pattern)
    data_nutrients = matched_columns.reindex(sorted(matched_columns.columns), axis=1)

    X = data_nutrients.values
    attributes_dummies = data_nutrients.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(X)
    nutrients_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)

    self.nutrients_normalized = nutrients_normalized.replace(np.nan,0)
    self.data_label = self.data[self.label]

  def create_embedding(self):
    embedding = self.nutrients_normalized.copy()
    embedding = embedding.replace(np.nan,0)

    self.label_encoder = LabelEncoder()
    self.X = embedding.values
    y = self.data_label.values
    self.y = self.label_encoder.fit_transform(y)

training_usda = Training("USDA")
training_vegan = Training("Vegan")
training_gluten = Training("Gluten")
training_vegcat = Training("Vegetarian-Vegan-Omni")
print(f"Vegan F1 Macro: {training_vegan.f1_macro:.7f}, F1 Micro: {training_vegan.f1_micro:.7f}")
print(f"Gluten F1 Macro: {training_gluten.f1_macro:.7f}, F1 Micro: {training_gluten.f1_micro:.7f}")
print(f"Vegetarian-Vegan-Omni F1 Macro: {training_vegcat.f1_macro:.7f}, F1 Micro: {training_vegcat.f1_micro:.7f}")
