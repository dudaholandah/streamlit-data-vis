import pandas as pd
import streamlit as st
from auxiliary_functions import *

class DataProcessing:

  def __init__(self):
    self.pre_processing()
  
  def pre_processing(self): 

    # prevent from create an empty label
    if st.session_state['label'] == st.session_state['created_label'] and st.session_state['label'] == "":
      st.error("Don't forget to name the label on the sidebar!", icon="🚨")
      st.session_state['global_vis_ready'] = False
      st.session_state['local_vis_ready'] = False
      return

    # if the data is null initialize state and skip function
    if len(st.session_state['selected_attributes']) < 2:
      st.session_state['global_vis_ready'] = False
      st.session_state['local_vis_ready'] = False
      return

    # define data
    if st.session_state['label'] in st.session_state['selected_attributes']: st.session_state['selected_attributes'].remove(st.session_state['label']) 
    df = st.session_state['df'].copy()
    label = st.session_state['label']
    selected_attributes = st.session_state['selected_attributes']
    legend_attributes = st.session_state['legend_attributes']
    inspection_attr = 'Ingredients'
    
    # dataframes to be used
    data_legend = df[legend_attributes].copy()
    data_attr = df[selected_attributes].copy()
    data_label = df[label].copy()

    # define type
    type_of_columns = {column: define_type(df[column]) for column in df.columns}
    filtered_type_of_columns = {column: column_type for column, column_type in type_of_columns.items() if column in selected_attributes}
    numeric_types = [column for column, column_type in filtered_type_of_columns.items() if column_type == "Numeric"]
    string_types = [column for column, column_type in filtered_type_of_columns.items() if column_type != "Numeric" and column_type != "Unknown type"]
    
    # pre-processing numbers into a [0,1] range
    data_attr_normalized = pre_processing_numbers(df[df.columns.intersection(numeric_types)])
    # pre-processing strings into one-hot-encoding
    data_attr_normalized = data_attr_normalized.join(pre_processing_strings(df[df.columns.intersection(string_types)]))

    # pre-processing label
    for i in range(data_label.shape[0]):
      data_label[i] = data_label[i].rstrip()

    # split data
    X = data_attr_normalized.values

    # join all data
    merge = lambda df1, df2, df3 : pd.merge(pd.merge(df1, df2, left_index=True, right_index=True), df3, left_index=True, right_index=True)
    selected_data = merge(data_attr, data_label, data_legend)
    selected_data_normalized = merge(data_attr_normalized, data_label, data_legend)

    if inspection_attr not in selected_attributes and \
       inspection_attr not in legend_attributes and \
       inspection_attr != label:
      data_inspect = df[inspection_attr].copy()
      selected_data = selected_data.join(data_inspect)
      selected_data_normalized = selected_data_normalized.join(data_inspect)

    st.session_state['selected_data'] = selected_data
    st.session_state['selected_data_normalized'] = selected_data_normalized
    st.session_state['X'] = X
    st.session_state['global_vis_ready'] = True

  
  