import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
from unidecode import unidecode
import re
from itertools import combinations
from collections import defaultdict
from pyvis.network import Network
import networkx as nx
import random
import streamlit as st
from auxiliary_functions import *
import umap

class Visualizations:

  def __init__(self):
    self.plotly_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3    
  
  def create_scatterplot(self, X, y, title="", xaxis="", yaxis=""):
    label = st.session_state['label']
    st.session_state['selected_data']['x'] = X
    st.session_state['selected_data']['y'] = y

    selected_attributes = st.session_state['selected_attributes']
    legend_attributes = st.session_state['legend_attributes']

    hover_data = {col: True for col in (selected_attributes + legend_attributes)}
    hover_data['x'] = False
    hover_data['y'] = False

    sorted_categories = sorted(set(st.session_state['selected_data'][label]))

    fig = px.scatter(
      st.session_state['selected_data'], 
      x='x', 
      y='y', 
      color=label,
      template='plotly_white',
      color_discrete_sequence=self.plotly_colors,
      category_orders={ label: sorted_categories}, 
      height=650,
      hover_data=hover_data
    )
    
    fig.update_layout(xaxis_title=xaxis,
      yaxis_title=yaxis,
      title=title,
      xaxis=dict(ticks="outside", mirror=True, showline=True),
      yaxis=dict(ticks="outside", mirror=True, showline=True),
      margin=dict(l=0, r=0, b=50, t=50),
    )
    
    fig.update_xaxes(ticklen=0, showticklabels=False)
    fig.update_yaxes(ticklen=0, showticklabels=False)
    
    return fig
  
  def scatterplot_pca(self):

    sklearn_pca = sklearnPCA(n_components=2)
    X_pca = sklearn_pca.fit_transform(st.session_state['X'])

    X_pca[1:4, :]
    X = X_pca[:,0]
    y = X_pca[:,1]

    variance = sklearn_pca.explained_variance_ratio_
    scatterplot = self.create_scatterplot(X, y, "PCA Visualization", "Principal Component 1", "Principal Component 2")

    return scatterplot, variance
  
  def scatterplot_tsne(self):
    tsne = TSNE(n_components=2,perplexity=5,learning_rate=350,metric='euclidean', init='pca')
    X_tsne = tsne.fit_transform(st.session_state['X'])

    X_tsne[1:4, :]
    X = X_tsne[:,0]
    y = X_tsne[:,1]
    
    return self.create_scatterplot(X, y, "t-SNE Visualization")
  
  def scatterplot_umap(self):

    reducer = umap.UMAP(n_components=2,n_neighbors=10, min_dist=0.1,metric='euclidean', random_state=0)
    X_umap = reducer.fit_transform(st.session_state['X'])

    X_umap[1:4, :]
    X = X_umap[:,0]
    y = X_umap[:,1]
    
    return self.create_scatterplot(X, y, "UMAP Visualization")
  
  def parallel_coordinates(self):


    # define data
    entire_data = st.session_state['selected_data'].copy()
    filtered_data = st.session_state['df_filtered'].copy()
    label = st.session_state['label']
    type_of_columns = {column: define_type(entire_data[column]) for column in entire_data.columns}
    attributes = [column for column in st.session_state['selected_attributes'] if type_of_columns[column] == "Numeric"]

    ## calculate the mean for each category
    if (st.session_state['parallel_coordinates_option'] == "Display Mean Values"):
      filtered_data = filtered_data.groupby(label)[attributes].mean().reset_index()

    ## mapping only the categories selected
    sorted_categories = sorted(filtered_data[label].unique(), reverse=True)
    category_map = {category: i for i, category in enumerate(sorted_categories)}
    filtered_data['color_value'] = filtered_data[label].map(category_map)
    num_categories = len(sorted_categories)

    ## coloring all categories on the dataset
    all_categories = sorted(entire_data[label].unique(), reverse=True)
    reversed_colors = self.plotly_colors[:len(all_categories)][::-1]
    num_colors = len(reversed_colors)
    color_map = {category: reversed_colors[i % num_colors] for i, category in enumerate(all_categories)}

    ## for each category select the corresponding color (based on the original dataset)
    color_continuous_scale = []
    for i, category in enumerate(sorted_categories):
      start = i / num_categories
      end = (i + 1) / num_categories
      color = color_map[category]

      color_continuous_scale.append((start, color))
      color_continuous_scale.append((end, color))

    fig = px.parallel_coordinates(filtered_data, 
                                  dimensions=attributes, 
                                  color='color_value', 
                                  color_continuous_scale=color_continuous_scale, 
                                  labels={'color_value': label}, height=550)
            
    fig.update_layout(
      coloraxis_colorbar=dict(
        tickvals=[i for i in range(num_categories)],
        ticktext=sorted_categories,
        lenmode="pixels", len=300,
      ),
      margin=dict(l=40, r=0, t=50, b=50)
    )

    return fig
  
  def create_graph_network(self):

    data = st.session_state['df_filtered']
    column = 'Ingredients'
    edge_count = st.session_state['edge_count']
    times = st.session_state['times']

    if column not in st.session_state['df_filtered'].columns:
      return st.error("You need a column named **Ingredients** to be able to see this Visualization", icon="🚨")

    data_ingredients = [x.split(',') for x in data[column]]
    ignore_ingredients = st.session_state['ingredients_to_ignore']

    # pre processando ingredientes
    for lst in data_ingredients:
      for idx, each in enumerate(lst):
        lst[idx] = pre_process_string(each)

    # # removendo ingredientes que nao quero
    for lst in data_ingredients:
      for idx, each in enumerate(lst):
        if each in ignore_ingredients:
          lst.pop(idx)

    # dicionário para armazenar a contagem de pares de ingredientes
    ingredient_pairs_count = defaultdict(int)
    ingredient_pairs_count_reduced = defaultdict(int)

    # contando os pares de ingredientes
    for array in data_ingredients:
      for pair in combinations(array, 2):
        ingredient_pairs_count[pair] += 1

    # contagem de pares de ingredientes
    for pair, count in ingredient_pairs_count.items():
      if count >= times:
        ingredient_pairs_count_reduced[pair] = count

    # nodes e edges
    nodes = list(set(ingredient for pair in ingredient_pairs_count_reduced for ingredient in pair))
    edges = [(pair[0], pair[1], count) for pair, count in ingredient_pairs_count_reduced.items()]

    # xy
    mn = 1
    mx = len(nodes)

    node_x = []
    node_y = []
    for _ in nodes:
      node_x.append(random.randint(mn, mx)) 
      node_y.append(random.randint(mn, mx)) 

    # create network
    net = Network(height="500px")
    net_to_download = Network(height="850px")

    for src, dest, count in edges:
      
      # size of node
      num_src = 1
      num_dest = 1
      for e in edges:
        if src in e: num_src += 1
        if dest in e: num_dest += 1

      net.add_node(src, size=num_src)
      net.add_node(dest, size=num_dest)
      net_to_download.add_node(src, size=num_src)
      net_to_download.add_node(dest, size=num_dest)

      if edge_count:
        net.add_edge(src, dest, value=count)
        net_to_download.add_edge(src, dest, value=count)
      else:
        net.add_edge(src, dest)
        net_to_download.add_edge(src, dest)

    net.save_graph("data/graph_to_display.html")
    net_to_download.save_graph("data/graph_to_download.html")
    html_file_display = open("data/graph_to_display.html", 'r', encoding='utf-8')
    html_file_download = open("data/graph_to_download.html", 'r', encoding='utf-8')
    source_code_display = html_file_display.read() 
    source_code_download = html_file_download.read() 
    return source_code_display, source_code_download