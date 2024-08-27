import pandas as pd
from upload import *
from data_processing import *
from auxiliary_functions import *
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

def filter_points():
  category_points = st.multiselect("Select the points based on their Category (color)", sorted(st.session_state['df'][st.session_state.label].unique()))
  if category_points:
    selected_points_by_category = st.session_state['df'][st.session_state['df'][st.session_state.label].isin(category_points)]
  else:
    selected_points_by_category = pd.DataFrame()

  ingredients_points = st.multiselect("Select the points based on their Ingredients", 
                                      [] if 'Ingredients' not in st.session_state['df'].columns
                                      else vocab_list(st.session_state['df']['Ingredients']), 
                                      placeholder="Type or search the ingredient")
  if ingredients_points:
    copy_of_df = st.session_state['df'].copy()
    ## this way we select a substring
    # copy_of_df['Processed_Ingredients'] = st.session_state['df']['Ingredients'].apply(lambda x: pre_process_string(x))
    # selected_points_by_ingredients = copy_of_df[copy_of_df['Processed_Ingredients']
    #                                  .apply(lambda x: any(ingredient in x for ingredient in ingredients_points))]
    # copy_of_df.drop('Processed_Ingredients', axis=1, inplace=True)
    
    ## this way we select the exact word split by comma
    filtered_rows = []
    for i, ingredients in enumerate(copy_of_df['Ingredients']):
      list_of_ingredients = ingredients.split(',')
      for word in list_of_ingredients:
        processed_word = pre_process_string(word)
        if processed_word in ingredients_points:
          filtered_rows.append(copy_of_df.iloc[i])
          break

    selected_points_by_ingredients = pd.DataFrame(filtered_rows)    
  else:
    selected_points_by_ingredients = pd.DataFrame()

  ## this way we select the inner join of category and ingredients selection
  # if selected_points_by_category.empty:
  #   selected_points_multiselect = selected_points_by_ingredients.copy()
  # elif selected_points_by_ingredients.empty:
  #   selected_points_multiselect = selected_points_by_category.copy()
  # else:
  #   selected_points_multiselect = selected_points_by_category[selected_points_by_category.index.isin(selected_points_by_ingredients.index)]
  
  ## this way we select the outter join of category and ingredients selection
  selected_points_multiselect = pd.concat([selected_points_by_category, selected_points_by_ingredients])
  return selected_points_multiselect

def render_global_visualization():

  if st.session_state['global_vis_ready'] == False:
    return
    
  visualizations = Visualizations()

  # define point based visualization 
  if st.session_state['scatterplot_option'] == "PCA":
    scatterplot_vis, variance = visualizations.scatterplot_pca()
  else:
    scatterplot_vis = visualizations.scatterplot_tsne()

  st.subheader("Visualization 1: Point based visualization") 
  st.caption("Select a sample of points to be analyzed")

  # plot PCA metrics
  if st.session_state['scatterplot_option'] == "PCA":
    col1, col2, col3 = st.columns(3)
    col1.metric(label='Total Explained Variance', value=f"{variance.sum()* 100:.2f}%")
    col2.metric(label='Principal Component 1 Explained Variance', value=f"{variance[0]* 100:.2f}%")
    col3.metric(label='Principal Component 2 Explained Variance', value=f"{variance[1]* 100:.2f}%")

  # select value for label
  if st.session_state['label'] == st.session_state['created_label']:
    is_creating_label = st.checkbox("Do you want to select the points to label them?")
    if is_creating_label:
      st.text_input('Type the next value of the label you created:', key='new_label')
  else:
    is_creating_label = False
  
  # plot and filter points in the visualization 
  selected_points = plotly_events(scatterplot_vis, select_event=True, override_height=650, key='selected_points')
  points_dataframe = selected_points_to_dataframe(selected_points)

  # filter points based on their category or ingredients
  selected_points_multiselect = filter_points()

  update_state(points_dataframe if not points_dataframe.empty else selected_points_multiselect, is_creating_label)


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
      st.subheader("Visualization 3: Graph based visualization") 
      graph_display, graph_download = visualizations.create_graph_network()
      components.html(graph_display, height=520)
      st.sidebar.download_button(label='Download Network Graph',
                      data=graph_download,
                      file_name='graph_network_result.html')
    except Exception as e: print(f"An error occurred: {e}")

  st.subheader("Selected points")
  st.write(st.session_state['df_filtered'])

def render_metrics():

  col1, col2 = st.columns(2)

  if st.session_state['global_vis_ready']:
    col1.metric(label='Number of dataset instances', value=len(st.session_state['df']))

  if st.session_state['local_vis_ready']:
    col2.metric(label='Number of sample instances', value=len(st.session_state['df_filtered']))


def render_interface():
  Upload()

def pre_processing():
  DataProcessing()

def main():
  render_interface()
  pre_processing()
  render_metrics()
  render_global_visualization()
  render_local_visualizations()

if __name__ == "__main__":
  initialize_state()
  main()