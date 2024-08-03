import streamlit as st
from auxiliary_functions import filter_dataframe

def initialize_state():
  if 'selected_points_query' not in st.session_state:
    st.session_state['selected_points_query'] = []

  if 'global_vis_ready' not in st.session_state:
    st.session_state['global_vis_ready'] = False

  if 'local_vis_ready' not in st.session_state:
    st.session_state['local_vis_ready'] = False

  if 'created_label' not in st.session_state:
    st.session_state['created_label'] = 'New label' 

def clear_state():    
  st.session_state['selected_points_query'] = []
  st.session_state['global_vis_ready'] = False
  st.session_state['local_vis_ready'] = False

def update_state(current_query, is_creating_label):
  rerun = False
  if current_query != st.session_state['selected_points_query']:
    st.session_state['selected_points_query'] = current_query
    rerun = True
  if rerun:
    filter_dataframe(current_query, is_creating_label) 
    st.experimental_rerun()