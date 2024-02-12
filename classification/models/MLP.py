import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import keras_tuner
import numpy as np

class MLPModel:
  def __init__(self, X, y, name, path, encoder, model=None):
    # train = 70%, test = 15%, valid = 15%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val
    self.X_test = X_test
    self.y_test = y_test
    self.name = name
    self.path = path
    self.encoder = encoder
    self.model = model    

  def hyperparameter_tuning(self):

    tuner = keras_tuner.RandomSearch(
      lambda hp: self.build_model(hp, len(self.X_train[0]), len(self.encoder.classes_)),
      objective='val_loss',
      project_name=self.name + "_project",
      directory=self.path
    )

    tuner.search(self.X_train, to_categorical(self.y_train), epochs=50, validation_data=(self.X_val, to_categorical(self.y_val)))
    best_model = tuner.get_best_models()[0]
    best_model.build()

    self.model = best_model
    self.model.save(self.path + self.name)

    return best_model

  def build_model(self, hp, input_shape, output_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Choice('units', [64, 128, 256]), input_shape=(input_shape,), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.5, 0.8, step=0.1)))
    model.add(tf.keras.layers.Dense(hp.Choice('units', [16, 32, 64]), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.5, 0.8, step=0.1)))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )
    
    return model

  def evaluate_model(self):
    y_prob = self.model.predict(self.X_test)
    y_pred = np.argmax(y_prob, axis=1)

    f1_macro = f1_score(self.y_test, y_pred, average='macro')
    f1_micro = f1_score(self.y_test, y_pred, average='micro')

    return f1_macro, f1_micro