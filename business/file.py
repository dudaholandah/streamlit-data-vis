import pandas as pd
import openpyxl

from sklearn import preprocessing
import pandas as pd
import numpy as np
from unidecode import unidecode

class File:
  def __init__(self, frontend):
    self.frontend = frontend

  def load(self):
    file_uploaded = self.frontend.file_uploader("Upload a file",type=(["xlsx","xls"]))
  
    if file_uploaded is not None:
      wb = openpyxl.load_workbook(file_uploaded, read_only=True)
      sheetname = self.frontend.selectbox("Select the sheetname", wb.sheetnames)
      df = pd.read_excel(file_uploaded, sheet_name=sheetname)
      self.df = df
    else:
      self.frontend.caption("Demo file has been uploaded. If you would like to see another file, feel free to upload.")
      df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos")
      self.df = df 

  def select_label(self):
    self.label = self.frontend.selectbox("Select the label", self.df.columns)

  def select_attributes(self):
    self.attributes = self.frontend.multiselect("Select the attributes", self.df.columns)

  def select_legend(self):
    self.legend = self.frontend.multiselect("Add attributes to the legend", [col for col in self.df.columns if (col not in self.attributes) and (col != self.label)])

  def select_inspection_attr(self):
    self.inspection_attr = self.frontend.selectbox("Select the attribute to be analyzed in the Neural Network and Wordcloud", self.df.columns)
    self.multip = self.frontend.slider("Select the size of the nodes", 1, 5)
    self.times = self.frontend.slider("Select the frequency of connections", 1, 10, 5)
    
  def pre_processing_numbers(self, data):
    attributes_dummies = data.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(data.values)
    data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
    data_normalized = data_normalized.replace(np.nan,0)
    return data_normalized

  ### todo: pre processar strings / listas ...
  def pre_processing(self):    
    # define data
    self.data_attr = self.df[self.attributes]
    self.data_label = self.df[self.label]
    self.data_legend = self.df[self.legend]

    # pre-processing numbers into a [0,1] range
    self.data_attr_normalized = self.pre_processing_numbers(self.data_attr)

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

  def filter_dataframe(self, selected_points):
    data = self.selected_data_normalized
    selected_rows = []
    selected_idx = set()

    for i in data.index:
      for pt in selected_points:
        if data['x'][i] == pt['x'] and data['y'][i] == pt['y'] and i not in selected_idx:
          selected_idx.add(i)
          selected_rows.append(pd.DataFrame([self.df.iloc[i]]))

    df_filtred = pd.concat(selected_rows, ignore_index=True)
    self.df_filtred = df_filtred 
  

  
