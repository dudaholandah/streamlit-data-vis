import pandas as pd
import openpyxl

from sklearn import preprocessing
import numpy as np
from unidecode import unidecode
import re

class File:
  def __init__(self, frontend):
    self.frontend = frontend

  def load(self):
    file_uploaded = self.frontend.file_uploader("Upload a file",type=(["xlsx","xls"]), key="file_uploaded")
  
    if file_uploaded is not None:
      wb = openpyxl.load_workbook(file_uploaded, read_only=True)
      sheetname = self.frontend.selectbox("Select the sheetname", wb.sheetnames, key="sheetname")
      df = pd.read_excel(file_uploaded, sheet_name=sheetname)
      self.df = df
    else:
      self.frontend.caption("Demo file has been uploaded. If you would like to see another file, feel free to upload.")
      df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos")
      self.df = df 

  def select_label(self):
    self.label = self.frontend.selectbox("Select the label", self.df.columns, key="label")

  def select_attributes(self):
    self.attributes = self.frontend.multiselect("Select the attributes", [col for col in self.df.columns], key="attributes")

  def select_legend(self):
    self.legend = self.frontend.multiselect("Add attributes to the legend", [col for col in self.df.columns if (col not in self.attributes) and (col != self.label)], key="legend")

  def select_inspection_attr(self):
    self.inspection_attr = self.frontend.selectbox("Select the attribute to be analyzed in the Neural Network and Wordcloud", self.df.columns, key="inspection_attr")
    self.multip = self.frontend.slider("Select the size of the nodes", 1, 5)
    self.times = self.frontend.slider("Select the frequency of connections", 1, 10, 5)
  
  def pre_processing_numbers(self, data):
    attributes_dummies = data.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(data.values)
    data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
    data_normalized = data_normalized.replace(np.nan,0)
    return data_normalized
  
  def pre_process(self, text):
    text = re.sub(r'[.,():%-]+', " ", text)
    text = re.sub(r'[\s]+', " ", text)
    text = unidecode(text.strip().lower())
    return text
  
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

  def pre_processing(self):    
    # define data
    if self.label in self.attributes: self.attributes.remove(self.label) 
    self.data_attr = self.df[self.attributes].copy()
    self.data_label = self.df[self.label].copy()
    self.data_legend = self.df[self.legend].copy()

    # define type
    self.type_of_columns = {column: self.define_type(self.df[column]) for column in self.df.columns}
    filtered_type_of_columns = {column: column_type for column, column_type in self.type_of_columns.items() if column in self.attributes}
    numeric_types = [column for column, column_type in filtered_type_of_columns.items() if column_type == "Numeric"]
    string_types = [column for column, column_type in filtered_type_of_columns.items() if column_type != "Numeric" and column_type != "Unknown type"]
    
    # pre-processing numbers into a [0,1] range
    self.data_attr_normalized = self.pre_processing_numbers(self.df[self.df.columns.intersection(numeric_types)])
    # pre-processing strings into one-hot-encoding
    self.data_attr_normalized = self.data_attr_normalized.join(self.pre_processing_strings(self.df[self.df.columns.intersection(string_types)]))

    # pre-processing label
    for i in range(self.data_label.shape[0]):
      self.data_label[i] = self.data_label[i].rstrip()

    # split data
    self.X = self.data_attr_normalized.values
    self.y = self.data_label.values

    # join all data
    merge = lambda df1, df2, df3 : pd.merge(pd.merge(df1, df2, left_index=True, right_index=True), df3, left_index=True, right_index=True)
    self.selected_data = merge(self.data_attr, self.data_label, self.data_legend)
    self.selected_data_normalized = merge(self.data_attr_normalized, self.data_label, self.data_legend)

    if self.inspection_attr not in self.attributes and self.inspection_attr != self.label:
      data_inspect = self.df[self.inspection_attr].copy()
      self.selected_data = self.selected_data.join(data_inspect)
      self.selected_data_normalized = self.selected_data_normalized.join(data_inspect)

  def filter_dataframe(self, selected_points):
    data = self.selected_data
    selected_rows = []
    selected_idx = set()

    for i in data.index:
      for pt in selected_points:
        if abs(data['x'][i] - pt['x']) < 1e-4 and abs(data['y'][i] - pt['y']) < 1e-4 and i not in selected_idx:
          selected_idx.add(i)
          selected_rows.append(pd.DataFrame([data.iloc[i]]))

    df_filtred = pd.concat(selected_rows, ignore_index=True)
    self.df_filtred = df_filtred