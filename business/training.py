import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlpmodel import MLPModel
from sklearn.decomposition import PCA
import os
import re

class Training:

  SEP = os.path.sep + 'data\\'
  DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + SEP

  def __init__(self, name):
    self.name = name
    if name == "Vegan":
      self.data = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos")
      self.label = "Classification"
      for i, row_label in enumerate(self.data[self.label]):
        self.data.at[i, self.label] = re.sub(r'[0-9]+', "", row_label).strip()
    elif name == "USDA":
      self.data = pd.read_excel("data/usda_condesed_categories.xlsx", sheet_name="Sheet1")
      self.label = "Category"
    elif name == "Gluten":
      self.data = pd.read_excel("data/gluten_dataset.xlsx", sheet_name="dataset")
      self.label = "Product"
    elif name == "Vegetarian-Vegan-Omni":
      self.data = pd.read_excel("data/veg_category_usda_dataset.xlsx", sheet_name="Sheet1")
      self.label = "vegCategory"

    self.pre_processing()

    mlp_model = MLPModel(self.pca_result, self.y, self.name + "_mlp", self.DIR_PATH, self.label_encoder) 
    mlp_model.hyperparameter_tuning()

    self.f1_macro, self.f1_micro = mlp_model.evaluate_model()
  
  def pre_processing(self):
    columns = [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]
    data_numeric = self.data[columns]
    data_label = self.data[self.label]

    ## normalize nutrients
    X = data_numeric.values
    attributes_dummies = data_numeric.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(X)
    data_nutrients_normalized = pd.DataFrame(xscaled, columns=attributes_dummies)
    data_nutrients_normalized = data_nutrients_normalized.replace(np.nan,0)

    ## crate embedding
    embedding_bow = data_nutrients_normalized
    embedding_bow = embedding_bow.replace(np.nan,0)
    X = embedding_bow.values
    y = data_label.values
    self.label_encoder = LabelEncoder()
    self.y = self.label_encoder.fit_transform(y)

    ## space reduction
    pca = PCA(n_components=8)
    self.pca_result = pca.fit_transform(X)

training_usda = Training("USDA")
training_vegan = Training("Vegan")
training_gluten = Training("Gluten")
training_vegcat = Training("Vegetarian-Vegan-Omni")
print(f"USDA F1 Macro: {training_usda.f1_macro:.7f}, F1 Micro: {training_usda.f1_micro:.7f}")
print(f"Vegan F1 Macro: {training_vegan.f1_macro:.7f}, F1 Micro: {training_vegan.f1_micro:.7f}")
print(f"Gluten F1 Macro: {training_gluten.f1_macro:.7f}, F1 Micro: {training_gluten.f1_micro:.7f}")
print(f"Vegetarian-Vegan-Omni F1 Macro: {training_vegcat.f1_macro:.7f}, F1 Micro: {training_vegcat.f1_micro:.7f}")
