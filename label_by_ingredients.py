import pandas as pd
from auxiliary_functions import *
from label_data import *

class LabelByIngredients:
  def __init__(self):
    # init
    self.df = st.session_state['df']
    # define columns to look at
    type_of_columns = {column: define_type(self.df[column]) for column in self.df.columns}
    self.string_type_columns = [column for column, column_type in type_of_columns.items() if column_type == "List of strings" or column_type == "Comma-separated string"]

  def classify(self, name):
    if name == "Vegetarian":
      return self.classify_vegetarian()
    elif name == "Gluten":
      return self.classify_gluten()
    elif name == "Lactose":
      return self.classify_lactose()

  def classify_gluten(self):   
    # init result
    self.df['Gluten Classification'] = 'Gluten Free'

    for column in self.string_type_columns:
      for idx, row in self.df.iterrows():
        ingredients_list = [ingredient.strip() for ingredient in row[column].split(',')]
        for ingredient in ingredients_list:
          ingredient = pre_process_string(ingredient)
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
          ingredient = pre_process_string(ingredient)
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
          ingredient = pre_process_string(ingredient)
          if ingredient in non_vegetarian_ingredients:
            self.df.at[idx, 'Vegetarian Classification'] = 'Non Vegetarian'

    return 'Vegetarian Classification'
        
        
      
