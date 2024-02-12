from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import numpy as np

class RFModel:
  def __init__(self, X, y, name, path, model=None):
    self.outer_splits = 10
    self.inner_splits = 3
    self.param_grid = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [10, 100, 200],
        'max_features': [1, 3, 5, 7],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [1, 2, 3]
    }

    self.X = X
    self.y = y
    self.model = model
    self.name = name
    self.path = path

  def train_and_evaluate(self):
    outer_cv = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=42)

    f1_micro_scores = []
    f1_macro_scores = []

    for train_index, test_index in outer_cv.split(self.X, self.y):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]

      rf_classifier = RandomForestClassifier()

      inner_cv = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=42)

      grid_search = GridSearchCV(estimator=rf_classifier, param_grid=self.param_grid, scoring='f1_micro', cv=inner_cv)
      grid_search.fit(X_train, y_train)

      best_rf_model = grid_search.best_estimator_
      if self.model == None: self.model = best_rf_model

      y_pred = best_rf_model.predict(X_test)
      f1_micro = f1_score(y_test, y_pred, average='micro')
      f1_macro = f1_score(y_test, y_pred, average='macro')
      f1_micro_scores.append(f1_micro)
      f1_macro_scores.append(f1_macro)

      y_pred_prev = self.model.predict(X_test)
      f1_micro_prev = f1_score(y_test, y_pred_prev, average='micro')

      if f1_micro > f1_micro_prev: self.model = best_rf_model

    mean_f1_micro = np.mean(f1_micro_scores)
    std_f1_micro = np.std(f1_micro_scores)

    mean_f1_macro = np.mean(f1_macro_scores)
    std_f1_macro = np.std(f1_macro_scores)

    pickle.dump(self.model, open(self.path + self.name + ".pkl", 'wb'))

    return mean_f1_macro, std_f1_macro, mean_f1_micro, std_f1_micro
