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
    st.sidebar.caption("Demo file has been upload. If you would like to see another file, feel free to upload.")
    df = pd.read_excel("data/vegan_dataset.xlsx", sheet_name="Veganos")
    st.sidebar.divider()
    return df
  
def select_label_and_attributes(df):
  label = st.sidebar.selectbox("Select the label", df.columns)
  st.sidebar.divider()
  attributes = st.sidebar.multiselect("Select the attributes", df.columns)
  return label, attributes

def load_and_process_data():
  df = upload_file()
  label, attributes = select_label_and_attributes(df)
  X, y, data = pre_processing_data(df, label, attributes)
  return df, data, X, y, label, attributes

def render_plotly_ui(df, data, X, y, label, attributes):

  col01, col02 = st.columns((2))
  with col01:
    try:
      fig_spmatrix = create_scatterplotmatrix(data, label, attributes)
      st.plotly_chart(fig_spmatrix)
    except Exception as e: print(f"An error occurred: {e}")

  with col02:
    try:
      fig_parallelcoord = create_parallelcoordinates(data, label, attributes)
      st.plotly_chart(fig_parallelcoord, theme="streamlit")
    except Exception as e: print(f"An error occurred: {e}")


  selected_rows = []
  selected_idx = set()
  fig_pca = pca(data, X, y, label, len(attributes))
  selected_points =  plotly_events(fig_pca, select_event=True, key="tsne_query")

  for i in data.index:
    for pt in selected_points:
      if data['x'][i] == pt['x'] and data['y'][i] == pt['y'] and i not in selected_idx:
        selected_idx.add(i)
        selected_rows.append(pd.DataFrame([df.iloc[i]]))  

  col11, col12 = st.columns((2))

  with col11:
    df_selected_points = pd.concat(selected_rows, ignore_index=True)  
    fig_pca = create_wordcloud(df_selected_points)      
    st.pyplot(fig_pca)

  with col12:  
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
  
def main():
  try:
    df, data, X, y, label, attributes = load_and_process_data()
    current_query = render_plotly_ui(df, data, X, y, label, attributes)
    update_state(current_query)
    st.button("Reset filters", on_click=reset_state_callback)
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  st.set_page_config(page_title="FoodVis", page_icon=":apple:", layout="wide")
  st.sidebar.title("Food Vis")

  initialize_state()
  main()