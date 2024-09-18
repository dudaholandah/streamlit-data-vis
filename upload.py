import pandas as pd
import openpyxl
import streamlit as st
from set_state import *
from label_by_ingredients import *
from streamlit_tags import st_tags_sidebar

class Upload:
  def __init__(self):
    self.load()
    st.sidebar.divider()
    self.select_label()
    st.sidebar.divider()
    self.select_attributes()
    # st.sidebar.divider()
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
      option = st.sidebar.radio("Or select a between the following datasets:", ["Vegan Food Dataset", "USDA Food and Nutrient Database"])
      if option == "Vegan Food Dataset":
        df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos") 
      else:
        df = pd.read_excel("data/usda_fndds.xlsx", sheet_name="Sheet1") 

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
    st.sidebar.write("**Category** to color the points:")
    label_category = st.sidebar.radio("Select how you would like to aggregate the products in the dataset:", 
                                      ['Dataset Column', 'Create Label'], key="choice")
    # 'Gluten', 'Vegetarian', 'Lactose'
    
    # select a label from the dataset to color the points
    if label_category == 'Dataset Column':
      # select the column to be the label
      st.sidebar.caption("You must have a column named *Category* for this visualization.")
      label = 'Category'
      # label = st.sidebar.selectbox("Select the column from the dataset to color the points:", 
      #                              st.session_state['df'].columns, key="column_label")
      # save on session state
      if ('label' not in st.session_state) or (st.session_state['label'] != label):
        st.session_state['label'] = label
    # self color the dataset
    elif label_category == 'Create Label':
      # create the name of the color to be created
      new_label = st.sidebar.text_input("Create a new label for your dataset:", value='New Label', placeholder='Type a new value')
      # save on session state
      if 'label' not in st.session_state or st.session_state['label'] != new_label:
        
        st.session_state['created_label'] = new_label
        st.session_state['label'] = new_label
        st.session_state['df'][new_label] = 'Unlabeled'
        clean_df_state()
      # create a csv to download
      download_df_csv()
    # else:
    #   label_by_ing = LabelByIngredients()
    #   label = label_by_ing.classify(label_category)
    #   if ('label' not in st.session_state) or (st.session_state['label'] != label):
    #     st.session_state['label'] = label
    #     clean_df_state()
    #   # create a csv to download
    #   download_df_csv()

  def select_attributes(self):
    st.sidebar.write("**Food Components** in the Visualizations:")
    st.sidebar.multiselect("Select the components you would like to consider in the analysis:", 
                           [col for col in st.session_state['df'].columns], key="selected_attributes")

  def select_legend(self):
    st.sidebar.multiselect("Select additional components for hover information:", 
                           [col for col in st.session_state['df'].columns 
                            if (col not in st.session_state['selected_attributes']) 
                            and (col != st.session_state['label'])], key="legend_attributes")

  def select_scatterplot(self):
    st.sidebar.write("For **Visualization 1**:")
    st.sidebar.radio("Select a dimension reduction option:", ["PCA", "UMAP"], 
                     horizontal=True, key='scatterplot_option')

  def select_parallel_coordinates(self):
    st.sidebar.write("For **Visualization 2**:")
    st.sidebar.radio("Select the display option:", ["Display All Selection", "Display Mean Values"], 
                     horizontal=True,  key='parallel_coordinates_option')

  def select_inspection_attr(self):
    st.sidebar.write("For **Visualization 3**:")
    st.sidebar.caption("You must have a column named *Ingredients* for this visualization.")

    st.sidebar.slider("Select the minimum **threshold** of connections:", 1, 10, 5, key="times")

    st.sidebar.multiselect("Select the **Ingredients** you don't want to see in the visualization:", 
                            [] if 'Ingredients' not in st.session_state['df'].columns
                            else vocab_list(st.session_state['df']['Ingredients']),
                            placeholder="Type or search the ingredient", 
                            key='ingredients_to_ignore')

