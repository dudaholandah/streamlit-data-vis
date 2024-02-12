from dashboard import *
import streamlit as st

st.set_page_config(page_title="FoodVis", page_icon=":apple:", layout="wide")
st.sidebar.title("Food Vis")

if __name__ == "__main__":

  initialize_state()

  try:
    df, data, X, y, choice, label, attributes, legend, network_tuple = load_and_process_data()
    current_query = render_plotly_ui(df, data, X, y, choice, label, attributes, legend, network_tuple)
    update_state(current_query)
    st.button("Reset filters", on_click=reset_state_callback)
  except Exception as e:
    print(f"An error occurred: {e}")