import streamlit as st
import pandas as pd
from auxiliary_functions import filter_dataframe

def initialize_state():
  if 'selected_points_query' not in st.session_state:
    st.session_state['selected_points_query'] = pd.DataFrame()

  if 'global_vis_ready' not in st.session_state:
    st.session_state['global_vis_ready'] = False

  if 'local_vis_ready' not in st.session_state:
    st.session_state['local_vis_ready'] = False

  if 'created_label' not in st.session_state:
    st.session_state['created_label'] = 'New Label'
  
  if 'count' not in st.session_state:
    st.session_state.count = 0

def clear_state():    
  st.session_state['selected_points_query'] = pd.DataFrame()
  st.session_state['global_vis_ready'] = False
  st.session_state['local_vis_ready'] = False

def update_state(current_query, is_creating_label):
  rerun = False

  if not current_query.equals(st.session_state['selected_points_query']):
    st.session_state['selected_points_query'] = current_query.copy()
    rerun = True
  if rerun:
    filter_dataframe(current_query, is_creating_label) 
    st.experimental_rerun()

def clean_df_state():
  if len(st.session_state['df'].columns) >  len(st.session_state['uploaded_df'].columns) + 1:
    st.session_state['df'].drop(st.session_state['df'].columns[-2], axis=1, inplace=True)

def download_df_csv():
  download_df = st.session_state['df'].to_csv().encode('latin-1', 'ignore')
  st.sidebar.download_button(label='Download the labeled Dataset',
                    data=download_df,
                    file_name='new_labeled_dataframe.csv')