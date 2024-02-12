import pickle
import os
import pandas as pd
from models.MLP import MLPModel
from models.RF import RFModel
from models.SVM import SVMModel
from models.NB import NBModel
from training import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SEP = os.path.sep + 'data\\'
SRC_PATH = os.getcwd()
DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + SEP

df = pd.read_excel(f"{SRC_PATH}/data/veg_category_usda_dataset.xlsx", sheet_name='Sheet1')
data_nutrients = pd.DataFrame(df[['Carbohydrate', 'Protein', 'Fiber, total dietary', 'Sugars, total',  'Total Fat', 'Calcium', 'Sodium', 'Zinc']])
data_label = pd.DataFrame(df['vegCategory'])
nutrients_normalized = normalize_nutrients(data_nutrients)
X, y, label_encoder = create_embedding(nutrients_normalized, data_label)

saved_rf_model = pickle.load(open(DIR_PATH + "vegan_rf.pkl", 'rb'))
# print(saved_rf_model.classes_)

predictions = saved_rf_model.predict(X)

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
data = tsne.fit_transform(X)  # Reshape to (n_samples, 1) if predictions are 1D

data[1:4, :]
X = data[:,0]
y = data[:,1]

# Create a scatter plot for visualization
scatter = plt.scatter(X, y, c=predictions)  # Assuming 'y' is your true labels
plt.legend(handles=scatter.legend_elements()[0], labels=['DAIRY', 'EGG', 'FISH', 'MEAT', 'PORK', 'POULTRY'])
plt.title('t-SNE Visualization of RF Predictions')
plt.show()

# ['OMNI', 'VEGAN', 'VEGETARIAN']
# ['DAIRY', 'EGG', 'FISH', 'MEAT', 'PORK', 'POULTRY']
# ['Gluten Containing', 'Gluten Free']