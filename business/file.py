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

  ### todo: pre processar strings / listas ...
  def pre_processing(self):    
    # define data
    self.data_attr = self.df[self.attributes]
    self.data_label = self.df[self.label]

    # pre-processing numbers into a [0,1] range
    attributes_dummies = self.data_attr.columns
    normalize = preprocessing.MinMaxScaler()
    xscaled = normalize.fit_transform(self.data_attr.values)
    data_normalized = pd.DataFrame(xscaled,columns=attributes_dummies)
    data_normalized = data_normalized.replace(np.nan,0)
    self.data_attr_normalized = data_normalized

    # pre-processing label
    for i in range(self.data_label.shape[0]):
      self.data_label[i] = self.data_label[i].rstrip()

    # split data
    self.X = self.data_attr_normalized.values
    self.y = self.data_label.values

    # join all data
    self.all_data_normalized = self.data_attr_normalized.join(self.data_label)
  

  
