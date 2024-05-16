import streamlit as st
from streamlit_plotly_events import plotly_events
from business.file import File
from presentation.visualizations import Visualizations
import streamlit.components.v1 as components

class Page:

  def __init__(self):
    st.set_page_config(page_title="FoodVis", page_icon=":apple:", layout="wide")
    st.sidebar.title("Food Vis")

    st.markdown("""
      <style>
        .element-container {
          margin-top: -10px;
          padding-top: 0rem;
          padding-bottom: 0rem;
        }
      </style>
      """, unsafe_allow_html=True)

    self.file = File(st.sidebar)

  def display_sidebar(self):

    self.file.load()
    st.sidebar.divider()
    self.file.select_label()
    st.sidebar.divider()
    self.file.select_attributes()
    st.sidebar.divider()
    self.file.select_legend()
    st.sidebar.divider()
    self.file.select_inspection_attr()
    self.file.pre_processing()

  def display_visualizations(self):

    self.visualizations = Visualizations(st, self.file)

    selected_points = plotly_events(self.visualizations.scatterplot_pca(), select_event=True, key="selected_points", override_height=650)
    self.file.filter_dataframe(selected_points)

    col1, col2 = st.columns(2)
    with col1:
      try:
        graph = self.visualizations.create_graph_network()
        components.html(graph, height=520)
        st.sidebar.download_button(label='Download the Neural Network',
                        data=graph,
                        file_name='graph_neural_network.html')
      except Exception as e: print(f"An error occurred: {e}")
    
    with col2:
      try:
        st.plotly_chart(self.visualizations.parallel_coordinates(), use_container_width=True)
      except Exception as e: print(f"An error occurred: {e}")
