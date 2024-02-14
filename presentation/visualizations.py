from business.file import File
import plotly.express as px
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

  def scatterplot_matrix(self):
    data = self.file.selected_data
    label = self.file.label
    attributes = self.file.attributes
    
    set_label = set()

    for i in range(data.shape[0]):
      set_label.add(data[label][i])

    fig = px.scatter_matrix(data,
      dimensions=attributes,
      color=label,
      category_orders={label: sorted(set_label)},
      hover_data={col: True for col in self.file.legend}
    )

    fig.update_traces(diagonal_visible=False)
    return fig
  
  def pie_chart(self):

    data = self.file.selected_data.copy()
    label = self.file.label

    set_label = set()

    for i in range(data.shape[0]):
      set_label.add(data[label][i])
      data['index'] = i

    fig = px.pie(data, 
      values='index', 
      names=label, 
      category_orders={label: sorted(set_label)},
      hover_data={'index':False}
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

    fig = px.scatter(
      data, 
      x='x', 
      y='y', 
      color=label,
      template='plotly_white',
      category_orders={ label: sorted(set(self.file.y))}, 
      height=500,
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
    sklearn_pca = sklearnPCA(n_components= len(self.file.attributes))
    X_pca = sklearn_pca.fit_transform(self.file.X)

    X_pca[1:4, :]
    X = X_pca[:,0]
    y = X_pca[:,1]
    
    return self.create_scatterplot(X, y)
  
  def pre_process(self, text):
    text = re.sub(r'[.,():%-]+', " ", text)
    text = re.sub(r'[\s]+', " ", text)
    text = unidecode(text.strip().lower())
    return text

  def create_wordcloud(self):

    data = self.file.df_filtred
    column = self.file.inspection_attr

    comment_words = ''
    vocab = set()
    stopwords = set(STOPWORDS)

    for text in data[column]:
      tokens = text.split()
      for i in range(len(tokens)):
        vocab.add(self.pre_process(tokens[i]))

    for word in vocab:
      comment_words += " " + word + " "

    wordcloud = WordCloud(width = 800, height = 500,
                          background_color ='white',
                          stopwords = stopwords,
                          colormap='Set2',
                          min_font_size = 10).generate(comment_words)
    
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    return plt
  
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
    net = Network(height="450px")

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
