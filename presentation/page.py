import streamlit as st
from streamlit_plotly_events import plotly_events
from business.file import File
from presentation.visualizations import Visualizations

class Page:

  def __init__(self):
    st.set_page_config(page_title="FoodVis", page_icon=":apple:", layout="wide")
    st.sidebar.title("Food Vis")

    self.file = File(st.sidebar)

  def display_sidebar(self):
    self.file.load()
    st.sidebar.divider()
    self.file.select_label()
    st.sidebar.divider()
    self.file.select_attributes()
    self.file.pre_processing()

  def display_visualizations(self):

    self.visualizations = Visualizations(st, self.file)
    
    col1, col2 = st.columns(2)
    with col1:
      try:
        st.plotly_chart(self.visualizations.scatterplot_matrix())
      except Exception as e: print(f"An error occurred: {e}")
      
    with col2:
      try:
        st.plotly_chart(self.visualizations.pie_chart())
      except Exception as e: print(f"An error occurred: {e}")

    selected_points = plotly_events(self.visualizations.scatterplot_pca(), select_event=True,)