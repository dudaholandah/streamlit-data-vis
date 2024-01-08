import streamlit as st
from streamlit_plotly_events import plotly_events
import openpyxl
import pandas as pd
from auxiliary import * 
from visualizations import *

def upload_file():

  file_uploaded = st.sidebar.file_uploader("Upload a file",type=(["xlsx","xls"]))
  
  if file_uploaded is not None:
    wb = openpyxl.load_workbook(file_uploaded, read_only=True)
    sheetname = st.sidebar.selectbox("Select the sheetname", wb.sheetnames)
    df = pd.read_excel(file_uploaded, sheet_name=sheetname)
    st.sidebar.divider()
    return df
  else:
    st.sidebar.caption("Demo file has been uploaded. If you would like to see another file, feel free to upload.")
    df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos")
    st.sidebar.divider()
    return df
  
def select_label_and_attributes(df):
  label = st.sidebar.selectbox("Select the label", df.columns)
  st.sidebar.divider()
  attributes = st.sidebar.multiselect("Select the attributes", df.columns)
  return label, attributes

def select_neural_network_column(df):
  st.sidebar.divider()
  column = st.sidebar.selectbox("Select the attribute to be analyzed in the Neural Network and Wordcloud", df.columns)
  times = st.sidebar.number_input("Minimum times the pair of ingredients must repete in the dataset", min_value=1, value=5)
  multip = st.sidebar.number_input("Number of time to multiply the size of the nodes in the Neural Network", min_value=1, max_value=10, value=1)
  return column, times, multip

def select_legend(df, att, label):
  st.sidebar.divider()
  columns = st.sidebar.multiselect("Add attributes to the legend", [col for col in df.columns if (col not in att) and (col != label)])
  return columns

def load_and_process_data():
  df = upload_file()
  label, attributes = select_label_and_attributes(df)

  legend = select_legend(df, attributes, label)
  network_tuple = select_neural_network_column(df)

  X, y, data = pre_processing_data(df, label, attributes, legend)
  return df, data, X, y, label, attributes, legend, network_tuple

def render_plotly_ui(df, data, X, y, label, attributes, legend, network_tuple):

  col01, col02 = st.columns((2))
  with col01:
    try:
      fig_spmatrix = create_scatterplotmatrix(data, label, attributes, legend)
      st.plotly_chart(fig_spmatrix)
    except Exception as e: print(f"An error occurred: {e}")

  with col02:
    try:
      fig_piechart = create_piechart(data, label, attributes)
      st.plotly_chart(fig_piechart, theme="streamlit")
    except Exception as e: print(f"An error occurred: {e}")


  selected_rows = []
  selected_idx = set()
  fig_pca = pca(data, X, y, label, len(attributes), legend+attributes)
  selected_points =  plotly_events(fig_pca, select_event=True, key="tsne_query")

  for i in data.index:
    for pt in selected_points:
      if data['x'][i] == pt['x'] and data['y'][i] == pt['y'] and i not in selected_idx:
        selected_idx.add(i)
        selected_rows.append(pd.DataFrame([df.iloc[i]]))  

  col11, col12 = st.columns((2))

  with col11:
    df_selected_points = pd.concat(selected_rows, ignore_index=True)  
    fig_pca = create_wordcloud(df_selected_points, network_tuple[0])      
    st.pyplot(fig_pca)

  with col12:  
    try:
      create_graph_network(df_selected_points, network_tuple)
      import streamlit.components.v1 as components
      html_file = open("data/graph.html", 'r', encoding='utf-8')
      source_code = html_file.read() 
      components.html(source_code, height=500)
    except:
      st.write(df_selected_points)
    

  return selected_idx

def initialize_state():
  if 'tsne_query' not in st.session_state:
    st.session_state.tsne_query = set()

  if "counter" not in st.session_state:
    st.session_state.counter = 0

def update_state(current_query):
  rerun = False
  if current_query - st.session_state.tsne_query:
    st.session_state.tsne_query = current_query
    rerun = True

  if rerun:
    st.experimental_rerun()

def reset_state_callback():
    st.session_state.counter = 1 + st.session_state.counter
    st.session_state["tsne_query"] = set()