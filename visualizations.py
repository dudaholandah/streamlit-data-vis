import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
from auxiliary import * 
# pares de ingredients
from itertools import combinations
from collections import defaultdict
# graph network
from pyvis.network import Network
import networkx as nx
import random

def create_scatterplot(data, label, y_label, X, y):
  data['x'] = X
  data['y'] = y

  fig = px.scatter(
    data, 
    x='x', 
    y='y', 
    color=label,
    template='plotly_white',
    category_orders={ label: sorted(set(y_label))}, 
    height=500 
  )
  
  fig.update_layout(xaxis_title="",
    yaxis_title="",
    xaxis=dict(ticks="outside", mirror=True, showline=True),
    yaxis=dict(ticks="outside", mirror=True, showline=True)
  )
  
  fig.update_xaxes(ticklen=0, showticklabels=False)
  fig.update_yaxes(ticklen=0, showticklabels=False)
  
  return fig

def tsne(data, X, y, label):
  tsne = TSNE(n_components=2,perplexity=5,learning_rate=350,metric='euclidean', init='pca')
  X_tsne = tsne.fit_transform(X)
  X_tsne, y_tsne = get_x_y(X_tsne)

  return create_scatterplot(data, label, y, X_tsne, y_tsne)

def pca(data, X, y, label, n_components):
  sklearn_pca = sklearnPCA(n_components=n_components)
  X_pca = sklearn_pca.fit_transform(X)
  X_pca, y_pca = get_x_y(X_pca)
  
  return create_scatterplot(data, label, y, X_pca, y_pca)

def create_wordcloud(data):

  comment_words = ''
  vocab = set()
  stopwords = set(STOPWORDS)

  for col in data:
    if data[col].dtype == "object":
      for text in data[col]:
        tokens = text.split()
        for i in range(len(tokens)):
          vocab.add(pre_process(tokens[i]))


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

def create_scatterplotmatrix(data, label, att):

  set_label = set()

  for i in range(data.shape[0]):
    set_label.add(data[label][i])

  fig = px.scatter_matrix(data,
    dimensions=att,
    color=label,
    category_orders={label: sorted(set_label)}
  )

  fig.update_traces(diagonal_visible=False)
  return fig

def create_parallelcoordinates(data, label, attributes):

  df = data.copy()
  set_label = set()

  for i in range(df.shape[0]):
    set_label.add(df[label][i])
    df['index'] = i

  fig = px.pie(df, 
    values='index', 
    names=label, 
    category_orders={label: sorted(set_label)}
  )

  return fig

def create_graph_network(data):
  data_ingredients = [x.split(',') for x in data['Ingredients']]
  ignore_ingredients = ['water', 'salt']

  # pre processando ingredientes
  for lst in data_ingredients:
    for idx, each in enumerate(lst):
      lst[idx] = pre_process(each)

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
    if count > 5:
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

    net.add_node(src, size=num_src)
    net.add_node(dest, size=num_dest)
    net.add_edge(src, dest)

  net.save_graph("data/graph.html")
  
