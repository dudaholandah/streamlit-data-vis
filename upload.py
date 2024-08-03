import pandas as pd
import openpyxl
import streamlit as st
from set_state import *
from label_by_ingredients import *

class Upload:
  def __init__(self):
    self.load()
    st.sidebar.divider()
    self.select_label()
    st.sidebar.divider()
    self.select_attributes()
    st.sidebar.divider()
    self.select_legend()
    st.sidebar.divider()
    self.select_scatterplot()
    st.sidebar.divider()
    self.select_parallel_coordinates()
    st.sidebar.divider()
    self.select_inspection_attr()

  def load(self):

    # upload file
    file_uploaded = st.sidebar.file_uploader("Upload a file",type=(["xlsx","xls"]), key="file_uploaded")
  
    if file_uploaded is not None:
      wb = openpyxl.load_workbook(file_uploaded, read_only=True)
      sheetname = st.sidebar.selectbox("Select the sheetname", wb.sheetnames, key="sheetname")
      df = pd.read_excel(file_uploaded, sheet_name=sheetname)
    else:
      st.sidebar.caption("Demo file has been uploaded. If you would like to see another file, feel free to upload.")
      df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos") 

    # save on session state
    if 'uploaded_df' not in st.session_state:
      st.session_state['uploaded_df'] = df.copy()
      st.session_state['df'] = df.copy()
      clear_state()
    elif not df.equals(st.session_state['uploaded_df']):
      st.session_state['uploaded_df'] = df.copy()
      st.session_state['df'] = df.copy()
      clear_state()

  def select_label(self):
    label_category = st.sidebar.radio("Select the label category to color the points:", ['Dataset Column', 'Create New Label', 'Gluten', 'Vegetarian', 'Lactose'], key="choice")
    
    # select a label from the dataset to color the points
    if label_category == 'Dataset Column':
      # select the column to be the label
      label = st.sidebar.selectbox("Select the column from the dataset to color the points:", 
                                   st.session_state['df'].columns, key="column_label")
      # save on session state
      if ('label' not in st.session_state) or (st.session_state['label'] != label):
        st.session_state['label'] = label
    # self color the dataset
    elif label_category == 'Create New Label':
      # create the name of the color to be created
      new_label = st.sidebar.text_input("Create a new label for your dataset:", key="created_label")
      # save on session state
      if ('label' not in st.session_state) or (st.session_state['label'] != new_label):
        st.session_state['label'] = new_label
        st.session_state['df'][new_label] = 'Unlabeled'
    else:
      label_by_ing = LabelByIngredients()
      label = label_by_ing.classify(label_category)
      if ('label' not in st.session_state) or (st.session_state['label'] != label):
        st.session_state['label'] = label



  def select_attributes(self):
    st.sidebar.multiselect("Select the **Food Components** you would like to see in the Visualizations:", 
                           [col for col in st.session_state['df'].columns], key="selected_attributes")

  def select_legend(self):
    st.sidebar.multiselect("Select additional the **Food Components** for hover information:", 
                           [col for col in st.session_state['df'].columns 
                            if (col not in st.session_state['selected_attributes']) 
                            and (col != st.session_state['label'])], key="legend_attributes")

  def select_scatterplot(self):
    st.sidebar.radio("Select a dimension reduction option for **Visualization 1**:", ["PCA", "t-SNE"], 
                     horizontal=True, key='scatterplot_option')

  def select_parallel_coordinates(self):
    st.sidebar.radio("Select the display option for **Visualization 2**:", ["Display All Selection", "Display Mean Values"], 
                     horizontal=True,  key='parallel_coordinates_option')

  def select_inspection_attr(self):
    st.sidebar.selectbox("Select the column from the dataset to be analyzed in **Visualization 3**:", 
                         st.session_state['df'].columns, key="inspection_attr")
    st.sidebar.slider("Select **node size** proportional to number of connections:", 1, 5, key="multip")
    st.sidebar.slider("Select the minimum **threshold** of connections:", 1, 10, 5, key="times")
