from presentation.page import *

if __name__ == "__main__":
  try:
    page = Page()
    page.display_sidebar()
    page.display_visualizations()
  except Exception as e:
    print(f"An error occurred: {e}")