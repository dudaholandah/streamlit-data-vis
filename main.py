import pandas as pd
from upload import *
from data_processing import *
from visualizations import *
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components
from set_state import *

# page configurations
st.set_page_config(page_title="FoodVis", page_icon=":apple:", layout="wide")
st.sidebar.title("FoodVis")
st.markdown("""
  <style>
    .element-container {
      margin-top: -10px;
      padding-top: 0rem;
      padding-bottom: 0rem;
    }
  </style>
  """, unsafe_allow_html=True)

def render_global_visualization():

  if st.session_state['global_vis_ready'] == False:
    return
    
  # define point based visualization
  visualizations = Visualizations()

  if st.session_state['scatterplot_option'] == "PCA":
    scatterplot_vis, variance = visualizations.scatterplot_pca()
  else:
    scatterplot_vis = visualizations.scatterplot_tsne()

  st.subheader("Visualization 1: Point based visualization") 
  st.caption("Select a sample of points to be analyzed")

  if st.session_state['scatterplot_option'] == "PCA":
    col1, col2, col3 = st.columns(3)
    col1.metric(label='Total Explained Variance', value=f"{variance.sum()* 100:.2f}%")
    col2.metric(label='Principal Component 1 Explained Variance', value=f"{variance[0]* 100:.2f}%")
    col3.metric(label='Principal Component 2 Explained Variance', value=f"{variance[1]* 100:.2f}%")

  if st.session_state['label'] == st.session_state['created_label']:
    st.text_input('Type the next value of the label you created:', key='new_label')
  
  # plot visualization filter points
  selected_points = plotly_events(scatterplot_vis, select_event=True, override_height=650, key='selected_points')  
  update_state(selected_points)


def render_local_visualizations():

  if st.session_state['local_vis_ready'] == False:
    return
  

  visualizations = Visualizations()
  
  # visualizations after the selection of points
  col1, col2 = st.columns(2)
  
  # parallel coordinates
  with col1:
    try:
      st.subheader("Visualization 2: Parallel Coordinates") 
      parallel_coordinates = visualizations.parallel_coordinates()
      st.plotly_chart(parallel_coordinates, use_container_width=True) 
    except Exception as e: print(f"An error occurred: {e}")
  # network graph
  with col2:
    try:
      st.subheader("Visualization 3: Graph Network") 
      graph_display, graph_download = visualizations.create_graph_network()
      components.html(graph_display, height=520)
      st.sidebar.download_button(label='Download Graph Network',
                      data=graph_download,
                      file_name='graph_network_result.html')
    except Exception as e: print(f"An error occurred: {e}")

  st.subheader("Selected points")
  st.write(st.session_state['df_filtered'])

def render_statistics():

  col1, col2 = st.columns(2)

  if st.session_state['global_vis_ready'] == False:
    return

  with col1:
    st.metric(label='Number of dataset instances', value=len(st.session_state['df']))

  if st.session_state['local_vis_ready'] == False:
    return

  with col2:
    st.metric(label='Number of sample instances', value=len(st.session_state['df_filtered']))


def render_interface():
  Upload()

def pre_processing():
  DataProcessing()

def main():
  render_interface()
  pre_processing()
  render_statistics()
  render_global_visualization()
  # update_state(current_query)
  # try:
  render_local_visualizations()
  # except Exception as e: print(f"An error occurred: {e}")

if __name__ == "__main__":
  initialize_state()
  main()