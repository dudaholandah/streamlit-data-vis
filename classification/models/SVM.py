from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score
import numpy as np
import pickle

class SVMModel:
  def __init__(self, X, y, name, path, model=None):
    self.outer_splits = 10
    self.inner_splits = 3
    self.param_grid = {
      'C': [0.1, 1, 10, 100, 1000],
      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
      'kernel': ['rbf']
    }

    self.X = X
    self.y = y
    self.name = name
    self.path = path
    self.model = model

  def train_and_evaluate(self):
    outer_cv = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=42)

    f1_micro_scores = []
    f1_macro_scores = []

    for train_index, test_index in outer_cv.split(self.X, self.y):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]

      classifier = svm.SVC()

      inner_cv = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=42)

      grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, scoring='f1_micro', cv=inner_cv)
      grid_search.fit(X_train, y_train)

      best_svm_model = grid_search.best_estimator_
      if self.model == None: self.model = best_svm_model

      y_pred = best_svm_model.predict(X_test)
      f1_micro = f1_score(y_test, y_pred, average='micro')
      f1_macro = f1_score(y_test, y_pred, average='macro')
      f1_micro_scores.append(f1_micro)
      f1_macro_scores.append(f1_macro)

      y_pred_prev = self.model.predict(X_test)
      f1_micro_prev = f1_score(y_test, y_pred_prev, average='micro')
      if f1_micro > f1_micro_prev: self.model = best_svm_model

    mean_f1_macro = np.mean(f1_macro_scores)
    std_f1_macro = np.std(f1_macro_scores)

    mean_f1_micro = np.mean(f1_micro_scores)
    std_f1_micro = np.std(f1_micro_scores)

    pickle.dump(self.model, open(self.path + self.name + ".pkl", 'wb'))

    return mean_f1_macro, std_f1_macro, mean_f1_micro, std_f1_micro
