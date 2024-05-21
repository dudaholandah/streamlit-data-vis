from business.file import File
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from unidecode import unidecode
import re
from itertools import combinations
from collections import defaultdict
from pyvis.network import Network
import networkx as nx
import random

class Visualizations:

  def __init__(self, st, file : File):
    self.frontend = st
    self.file = file
    self.plotly_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3      

  def parallel_coordinates(self):
    data_orig = self.file.selected_data.copy()
    data = self.file.df_filtred.copy()
    label = self.file.label
    attributes = [column for column in self.file.attributes if self.file.type_of_columns[column] == "Numeric"]

    ## calculate the mean for each category
    if (self.file.parallel_coordinates_option == "Mean"):
      data = data.groupby(label)[attributes].mean().reset_index()

    ## mapping only the categories selected
    sorted_categories = sorted(data[label].unique(), reverse=True)
    category_map = {category: i for i, category in enumerate(sorted_categories)}
    data['color_value'] = data[label].map(category_map)
    num_categories = len(sorted_categories)

    ## coloring all categories on the dataset
    all_categories = sorted(data_orig[label].unique(), reverse=True)
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

    fig = px.parallel_coordinates(data, 
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
      margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig
  
  def create_scatterplot(self, X, y):
    label = self.file.label
    data = self.file.selected_data
    data['x'] = X
    data['y'] = y

    hover_data = {col: True for col in (self.file.attributes + self.file.legend)}
    hover_data['x'] = False
    hover_data['y'] = False

    sorted_categories = sorted(set(self.file.y))

    print(sorted_categories)

    fig = px.scatter(
      data, 
      x='x', 
      y='y', 
      color=label,
      template='plotly_white',
      color_discrete_sequence=self.plotly_colors,
      category_orders={ label: sorted_categories}, 
      height=650,
      hover_data=hover_data
    )
    
    fig.update_layout(xaxis_title="",
      yaxis_title="",
      xaxis=dict(ticks="outside", mirror=True, showline=True),
      yaxis=dict(ticks="outside", mirror=True, showline=True)
    )
    
    fig.update_xaxes(ticklen=0, showticklabels=False)
    fig.update_yaxes(ticklen=0, showticklabels=False)
    
    return fig
  
  def scatterplot_pca(self):
    sklearn_pca = sklearnPCA(n_components=2)
    X_pca = sklearn_pca.fit_transform(self.file.X)

    X_pca[1:4, :]
    X = X_pca[:,0]
    y = X_pca[:,1]
    
    return self.create_scatterplot(X, y)
  
  def scatterplot_tsne(self):
    tsne = TSNE(n_components=2,perplexity=5,learning_rate=350,metric='euclidean', init='pca')
    X_tsne = tsne.fit_transform(self.file.X)

    X_tsne[1:4, :]
    X = X_tsne[:,0]
    y = X_tsne[:,1]
    
    return self.create_scatterplot(X, y)

  def pre_process(self, text):
    text = re.sub(r'[.,():%-]+', " ", text)
    text = re.sub(r'[\s]+', " ", text)
    text = unidecode(text.strip().lower())
    return text
  
  def create_graph_network(self):

    data = self.file.df_filtred
    column = self.file.inspection_attr
    times = self.file.times
    multip = self.file.multip

    data_ingredients = [x.split(',') for x in data[column]]
    ignore_ingredients = ['water', 'salt']

    # pre processando ingredientes
    for lst in data_ingredients:
      for idx, each in enumerate(lst):
        lst[idx] = self.pre_process(each)

    # removendo ingredientes que nao quero
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

    for src, dest, count in edges:
      
      # size of node
      num_src = 1
      num_dest = 1
      for e in edges:
        if src in e: num_src += 1
        if dest in e: num_dest += 1

      net.add_node(src, size=num_src*multip)
      net.add_node(dest, size=num_dest*multip)
      net.add_edge(src, dest)

    net.save_graph("data/graph.html")
    html_file = open("data/graph.html", 'r', encoding='utf-8')
    source_code = html_file.read() 
    return source_code
