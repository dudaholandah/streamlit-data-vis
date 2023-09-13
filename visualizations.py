import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
from auxiliary import * 

def create_scatterplot(data, label, y_label, X, y):
  data['x'] = X
  data['y'] = y

  fig = px.scatter(
    data, 
    x='x', 
    y='y', 
    color=label,
    template='plotly_white',
    category_orders={ label: sorted(set(y_label))} )
  
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
    if data[col].dtype != "float64":
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

