import streamlit as st
from physics import physics_page

st.write("Main page")

tab1, tab2, tab3 = st.tabs(["Physics", "Calculus", "Statistics"])

with tab1:
    physics_page()
    
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)