import streamlit as st
import pandas as pd
import pydeck as pdf
from urllib.error import URLError

st.set_page_config(page_title = "Mapping Demo", page_icon = "🌍")

st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.write(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)
