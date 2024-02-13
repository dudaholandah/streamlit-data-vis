from business.file import File
import plotly.express as px
from sklearn.decomposition import PCA as sklearnPCA

class Visualizations:

  def __init__(self, st, file : File):
    self.frontend = st
    self.file = file

  def scatterplot_matrix(self):
    data = self.file.all_data_normalized
    label = self.file.label
    attributes = self.file.attributes
    
    set_label = set()

    for i in range(data.shape[0]):
      set_label.add(data[label][i])

    fig = px.scatter_matrix(data,
      dimensions=attributes,
      color=label,
      category_orders={label: sorted(set_label)},
      # hover_data={col: True for col in legend}
    )

    fig.update_traces(diagonal_visible=False)
    return fig
  
  def pie_chart(self):

    data = self.file.all_data_normalized.copy()
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
    data = self.file.all_data_normalized.copy()
    data['x'] = X
    data['y'] = y

    hover_data = {col: True for col in self.file.attributes}
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
      # labels={'x': False, 'y': False}
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

