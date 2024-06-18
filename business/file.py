import pandas as pd
import openpyxl
from business.data_processing import DataProcessing
from business.classification import Classification

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
    label = self.frontend.radio("Select the label category to color the points:", ['Dataset Column', 'Gluten', 'Vegetarian', 'Lactose'], key="choice")
    if label == 'Dataset Column':
      self.label = self.frontend.selectbox("Select the column from the dataset to color the points:", self.df.columns, key="label")
    else:
      classification = Classification(self.df)
      self.label = classification.classify(label)

  def select_attributes(self):
    self.attributes = self.frontend.multiselect("Select the **Food Components** you would like to see in the Visualizations:", [col for col in self.df.columns], key="attributes")

  def select_legend(self):
    self.legend = self.frontend.multiselect("Select additional the **Food Components** for hover information:", [col for col in self.df.columns if (col not in self.attributes) and (col != self.label)], key="legend")

  def select_scatterplot(self):
    self.scatterplot_option = self.frontend.radio("Select a dimension reduction option for **Visualization 1**:", ["PCA", "t-SNE"], horizontal=True)

  def select_parallel_coordinates(self):
    self.parallel_coordinates_option = self.frontend.radio("Select the display option for **Visualization 3**:", ["Regular", "Mean"], horizontal=True)

  def select_inspection_attr(self):
    self.inspection_attr = self.frontend.selectbox("Select the column from the dataset to be analyzed in **Visualization 2**:", self.df.columns, key="inspection_attr")
    self.multip = self.frontend.slider("Select **node size** proportional to number of connections:", 1, 5)
    self.times = self.frontend.slider("Select the minimum **threshold** of connections:", 1, 10, 5)

  def pre_processing(self): 
    # define pre-processing class
    data_preprocessing = DataProcessing()

    # define data
    if self.label in self.attributes: self.attributes.remove(self.label) 
    self.data_attr = self.df[self.attributes].copy()
    self.data_label = self.df[self.label].copy()
    self.data_legend = self.df[self.legend].copy()

    # define type
    self.type_of_columns = {column: data_preprocessing.define_type(self.df[column]) for column in self.df.columns}
    filtered_type_of_columns = {column: column_type for column, column_type in self.type_of_columns.items() if column in self.attributes}
    numeric_types = [column for column, column_type in filtered_type_of_columns.items() if column_type == "Numeric"]
    string_types = [column for column, column_type in filtered_type_of_columns.items() if column_type != "Numeric" and column_type != "Unknown type"]
    
    # pre-processing numbers into a [0,1] range
    self.data_attr_normalized = data_preprocessing.pre_processing_numbers(self.df[self.df.columns.intersection(numeric_types)])
    # pre-processing strings into one-hot-encoding
    self.data_attr_normalized = self.data_attr_normalized.join(data_preprocessing.pre_processing_strings(self.df[self.df.columns.intersection(string_types)]))

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