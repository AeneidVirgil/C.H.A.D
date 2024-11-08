import streamlit as st

def physics_page():
    st.write("Physics Page")
    st.write(derivatives(1,2))
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)



def derivatives(x,y ):
    st.write(x * y)
    return x + y
