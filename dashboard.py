import streamlit as st
from streamlit_plotly_events import plotly_events
import openpyxl
import pandas as pd
from auxiliary import * 
from visualizations import *

'''
  upload an excel file
    a) chosen by user
    b) demo -> vegan dataset
  :return: the excel as pd.DataFrame
'''
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
  
'''
  select label from the uploaded file or 
  select label from the classifications results
  :param df: the file as pd.DataFrame
  :return: choice, label
'''
def select_label(df):
  # label = st.sidebar.selectbox("Select the label", df.columns)

  choice = st.sidebar.radio("Select the label from:", ["Uploaded File", "Classification Results"])

  if choice == "Uploaded File":
    label = st.sidebar.selectbox("Select the label", df.columns)
  else:
    label = st.sidebar.selectbox("Select the label", ["(1)Vegan/Vegetarian/Omni", "(2)Gluten Containing/Gluten Free", "(3)Dairy/Egg/Fish/Meat/Pork/Poultry"])

  return choice, label

def select_attributes(df):
  st.sidebar.divider()
  attributes = st.sidebar.multiselect("Select the attributes", df.columns)
  return attributes

'''
  select the attr to be analyzed in the Neural Network and WordCloud
  :param df: the file as pd.DataFrame
  :return: attr, times the pair of ingredients must repete, time to multiply the size of the nodes
'''
def select_neural_network_column(df):
  st.sidebar.divider()
  column = st.sidebar.selectbox("Select the attribute to be analyzed in the Neural Network and Wordcloud", df.columns)
  times = st.sidebar.number_input("Minimum times the pair of ingredients must repete in the dataset", min_value=1, value=5)
  multip = st.sidebar.number_input("Number of time to multiply the size of the nodes in the Neural Network", min_value=1, max_value=10, value=1)
  return column, times, multip

'''
  select new attr to the legends
  :param df: the file as pd.DataFrame
  :param att: the attr already chosen
  :param label: the chosen label
  :return: columns
'''
def select_legend(df, att, label):
  st.sidebar.divider()
  columns = st.sidebar.multiselect("Add attributes to the legend", [col for col in df.columns if (col not in att) and (col != label)])
  return columns

'''
  load and process the dataset
  - upload_file
  - select_label
  - select_attributes
  - select_legend
  - select_neural_network_column
  - pre_processing_data
  :return: original dataframe, pre processed dataframe, X, y, label, attr, legend, network_tuple
'''
def load_and_process_data():
  df = upload_file()

  choice, label = select_label(df)

  attributes = select_attributes(df)

  legend = select_legend(df, attributes, label)
  network_tuple = select_neural_network_column(df)

  if choice == "Uploaded File":
    X, y, data = pre_processing_data(df, label, attributes, legend)
  else:
    X, y, data = pre_processing_data(df, "", attributes, legend)

  return df, data, X, y, choice, label, attributes, legend, network_tuple

'''
  render the user interface
  - create_scatterplotmatrix
  - create_piechart
  - plotly_events (pca)
  - create_wordcloud
  - create_graph_network
  - download_button (graph_network)
  :param: original dataframe, pre processed dataframe, X, y, label, attr, legend, network_tuple
'''
def render_plotly_ui(df, data, X, y, choice, label, attributes, legend, network_tuple):

  if choice == "Classification Results":
    data, label = predict_label(data, X, label)

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

      st.sidebar.download_button(label='Download the Neural Network',
                        data=source_code,
                        file_name='graph_neural_network.html')
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