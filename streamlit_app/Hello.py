import streamlit as st
import time 

st.set_page_config(
    page_title = "Stock Prediction",
    page_icon = "💵"
)

st.header("Stock Prediction")
st.write('This is a website for Predicting Stock Market')# List items
list_items = ["GOOG", "META" ]

# Display the list using st.write
st.markdown("#### List of stocks:")
st.markdown("<ol>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ol>", unsafe_allow_html=True)
