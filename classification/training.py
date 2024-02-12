import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from models.MLP import MLPModel
from models.RF import RFModel
from models.SVM import SVMModel
from models.NB import NBModel
from unidecode import unidecode
import re

SEP = os.path.sep + 'data\\'
SRC_PATH = os.getcwd()
DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + SEP

def pre_process(text):
  text = re.sub(r'"', " ", text)
  text = re.sub(r'[.,():%-]+', " ", text)
  text = re.sub(r'[\s]+', " ", text)
  text = re.sub(r'[0-9]+', " ", text)
  text = unidecode(text.strip().lower())
  return text

def normalize_nutrients(data_nutrients):
  X = data_nutrients.values
  attributes_dummies = data_nutrients.columns

  normalize = preprocessing.MinMaxScaler()
  xscaled = normalize.fit_transform(X)

  nutrients_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
  nutrients_normalized = nutrients_normalized.replace(np.nan,0)

  return nutrients_normalized

def create_embedding(data_attr, data_label):
  embedding = data_attr.copy()
  embedding = embedding.replace(np.nan,0)

  label_encoder = LabelEncoder()
  X = embedding.values
  y = data_label.values
  y_int = label_encoder.fit_transform(y)

  return X, y_int, label_encoder

def gluten_dataset():
  df = pd.read_excel(f"{SRC_PATH}/data/gluten_dataset.xlsx", sheet_name='dataset')
  
  data_nutrients = pd.DataFrame(df[['Carbohydrates (g/100g)', 'Protein (g/100g)', 'Fiber (g/100g)', 'Sugar (g/100g)', 'Total fat (g/100g)', 'Calcium  (mg/100g)','Sodium (mg/100g)', 'Zinc  (mg/100g)']])
  data_label = df['Product']

  nutrients_normalized = normalize_nutrients(data_nutrients)

  X, y, label_encoder = create_embedding(nutrients_normalized, data_label)

  mlp_model = MLPModel(X, y, "gluten_mlp", DIR_PATH, label_encoder) 
  mlp_model.hyperparameter_tuning()
  mlp_stats = mlp_model.evaluate_model()
  
  rf_model = RFModel(X, y, "gluten_rf", DIR_PATH)
  rf_stats = rf_model.train_and_evaluate()

  svm_model = SVMModel(X, y, "gluten_svm", DIR_PATH)
  svm_stats = svm_model.train_and_evaluate()

  nb_model = NBModel(X, y, "gluten_nb", DIR_PATH)
  nb_stats = nb_model.train_and_evaluate()

  return mlp_stats, rf_stats, svm_stats, nb_stats

def veg_category_dataset():
  df = pd.read_excel(f"{SRC_PATH}/data/veg_category_usda_dataset.xlsx", sheet_name='Sheet1')
  
  data_nutrients = pd.DataFrame(df[['Carbohydrate', 'Protein', 'Fiber, total dietary', 'Sugars, total',  'Total Fat', 'Calcium', 'Sodium', 'Zinc']])
  data_label = pd.DataFrame(df['vegCategory'])

  nutrients_normalized = normalize_nutrients(data_nutrients)

  X, y, label_encoder = create_embedding(nutrients_normalized, data_label)

  mlp_model = MLPModel(X, y, "veg_category_mlp", DIR_PATH, label_encoder) 
  mlp_model.hyperparameter_tuning()
  mlp_stats = mlp_model.evaluate_model()
  
  rf_model = RFModel(X, y, "veg_category_rf", DIR_PATH)
  rf_stats = rf_model.train_and_evaluate()

  svm_model = SVMModel(X, y, "veg_category_svm", DIR_PATH)
  svm_stats = svm_model.train_and_evaluate()

  nb_model = NBModel(X, y, "veg_category_nb", DIR_PATH)
  nb_stats = nb_model.train_and_evaluate()

  return mlp_stats, rf_stats, svm_stats, nb_stats

def vegan_dataset():
  df = pd.read_excel(f"{SRC_PATH}/data/vegan_dataset.xlsx", sheet_name='Veganos')
  
  data_nutrients = pd.DataFrame(df[['Carbohydrate', 'Dietary Fiber', 'Proteins', 'Sugars', 'Total Fats', 'Calcium', 'Sodium', 'Zinc']])
  data_label = pd.DataFrame(df['Classification'])

  for i, label in enumerate(data_label['Classification']):
    data_label.at[i,'Classification'] = re.sub(r'[0-9]+', "", label).strip()

  nutrients_normalized = normalize_nutrients(data_nutrients)

  X, y, label_encoder = create_embedding(nutrients_normalized, data_label)

  mlp_model = MLPModel(X, y, "vegan_mlp", DIR_PATH, label_encoder) 
  mlp_model.hyperparameter_tuning()
  mlp_stats = mlp_model.evaluate_model()
  
  rf_model = RFModel(X, y, "vegan_rf", DIR_PATH)
  rf_stats = rf_model.train_and_evaluate()

  svm_model = SVMModel(X, y, "vegan_svm", DIR_PATH)
  svm_stats = svm_model.train_and_evaluate()

  nb_model = NBModel(X, y, "vegan_nb", DIR_PATH)
  nb_stats = nb_model.train_and_evaluate()

  return mlp_stats, rf_stats, svm_stats, nb_stats


# gluten_mlp_stats, gluten_rf_stats, gluten_svm_stats, gluten_nb_stats = gluten_dataset()
# print(f"MLP: {gluten_mlp_stats}")
# print(f"RF: {gluten_rf_stats}")
# print(f"SVM: {gluten_svm_stats}")
# print(f"NB: {gluten_nb_stats}")

# veg_category_mlp_stats, veg_category_rf_stats, veg_category_svm_stats, veg_category_nb_stats = veg_category_dataset()
# print(f"MLP: {veg_category_mlp_stats}")
# print(f"RF: {veg_category_rf_stats}")
# print(f"SVM: {veg_category_svm_stats}")
# print(f"NB: {veg_category_nb_stats}")

# vegan_mlp_stats, vegan_rf_stats, vegan_svm_stats, vegan_nb_stats = vegan_dataset()
# print(f"MLP: {vegan_mlp_stats}")
# print(f"RF: {vegan_rf_stats}")
# print(f"SVM: {vegan_svm_stats}")
# print(f"NB: {vegan_nb_stats}")