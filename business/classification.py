import pandas as pd
from business.data_processing import DataProcessing
from business.classification_data import *

class Classification:
  def __init__(self, df : pd.DataFrame):
    # init
    self.df = df
    self.data_preprocessing = DataProcessing()
    # define columns to look at
    type_of_columns = {column: self.data_preprocessing.define_type(self.df[column]) for column in self.df.columns}
    self.string_type_columns = [column for column, column_type in type_of_columns.items() if column_type == "List of strings" or column_type == "Comma-separated string"]


  def classify(self, name):
    match name:
      case "Vegetarian":
        return self.classify_vegetarian()
      case "Gluten":
        return self.classify_gluten()
      case "Lactose":
        return self.classify_lactose()
        

  def classify_gluten(self):   
    # init result
    self.df['Gluten Classification'] = 'Gluten Free'

    for column in self.string_type_columns:
      for idx, row in self.df.iterrows():
        ingredients_list = [ingredient.strip() for ingredient in row[column].split(',')]
        for ingredient in ingredients_list:
          ingredient = self.data_preprocessing.pre_process(ingredient)
          if ingredient in gluten_containing_ingredients:
            self.df.at[idx, 'Gluten Classification'] = 'Gluten Containing'
          else:
            separated_words = [word.strip() for word in ingredient.split(' ')]
            for word in separated_words:
              if word in gluten_containing_ingredients:
                self.df.at[idx, 'Gluten Classification'] = 'Gluten Containing'
    
    return 'Gluten Classification'

  def classify_lactose(self):
    # init result
    self.df['Lactose Classification'] = 'Lactose Free'

    for column in self.string_type_columns:
      for idx, row in self.df.iterrows():
        ingredients_list = [ingredient.strip() for ingredient in row[column].split(',')]
        for ingredient in ingredients_list:
          ingredient = self.data_preprocessing.pre_process(ingredient)
          if ingredient in lactose_containing_ingredients:
            self.df.at[idx, 'Lactose Classification'] = 'Lactose Containing'

    return 'Lactose Classification'
  
  def classify_vegetarian(self):
    # init result
    self.df['Vegetarian Classification'] = 'Vegetarian'

    for column in self.string_type_columns:
      for idx, row in self.df.iterrows():
        ingredients_list = [ingredient.strip() for ingredient in row[column].split(',')]
        for ingredient in ingredients_list:
          ingredient = self.data_preprocessing.pre_process(ingredient)
          if ingredient in non_vegetarian_ingredients:
            self.df.at[idx, 'Vegetarian Classification'] = 'Non Vegetarian'

    return 'Vegetarian Classification'
        
        
      