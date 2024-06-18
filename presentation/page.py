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
    self.file.select_scatterplot()
    st.sidebar.divider()
    self.file.select_parallel_coordinates()
    st.sidebar.divider()
    self.file.select_inspection_attr()
    self.file.pre_processing()

  def display_visualizations(self):

    self.visualizations = Visualizations(st, self.file)

    # scatter plot
    scatterplot_option = (self.visualizations.scatterplot_pca() if self.file.scatterplot_option == "PCA" else self.visualizations.scatterplot_tsne())
    st.subheader("Visualization 1: Scatter Plot") 
    selected_points = plotly_events(scatterplot_option, select_event=True, key="selected_points", override_height=650)
    self.file.filter_dataframe(selected_points)

    # visualizations after the selection of points
    col1, col2 = st.columns(2)

    # parallel coordinates
    with col1:
      try:
        st.subheader("Visualization 2: Parallel Coordinates") 
        st.plotly_chart(self.visualizations.parallel_coordinates(), use_container_width=True)
      except Exception as e: print(f"An error occurred: {e}")
    
    # network graph
    with col2:
      try:
        st.subheader("Visualization 3: Graph Network") 
        graph_display, graph_download = self.visualizations.create_graph_network()
        components.html(graph_display, height=520)
        st.sidebar.download_button(label='Download Graph Network',
                        data=graph_download,
                        file_name='graph_network_result.html')
      except Exception as e: print(f"An error occurred: {e}")
