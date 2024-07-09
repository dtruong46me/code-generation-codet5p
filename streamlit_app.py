
import streamlit as st
from src.model.codet5p import load_model

st.title("Python Code Generation")
st.caption("Project 2")

st.write("---")
user_input = st.text_input("Enter your input")

model = None
temperature = 0.0
top_k = 1
top_p = 0.0

with st.sidebar:
    st.header("Choose model:")
    checkpoint = "Dinosaur1812/codet5p-770m-lora"
    checkpoint = st.selectbox("Model", options=["", "Salesforce/codet5p-220m",
                                                 "Salesforce/codet5p-770m", 
                                                 "cincin2399/codet5-fine-tuned", 
                                                 "Dinosaur1812/codet5p-770m-lora"])

    model = load_model(checkpoint)
    model.origin_model = model.get_codet5p()

    st.header("Generation Config")
    temperature = st.slider("temperature", min_value=0.0, max_value=1.00, step=0.01, value=1.0)
    top_k = st.slider("top_k", min_value=1, max_value=100, value=50)
    top_p = st.slider("top_p", min_value=0.0, max_value=1.0, step=0.01, value=1.0)
    max_new_tokens = st.slider("max_new_tokens", min_value=1, max_value=512, value=256)
    min_new_tokens = st.slider("min_new_tokens", min_value=1, max_value=256, value=16)

if st.button("Submit"):
    if model is None:
        st.warning("Choose model before generate!")

    else:
        st.info("Generating input...")
        generated_output = model.generate(user_input, temperature=temperature, top_k=top_k, top_p=top_p, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)
        
        st.write("---")
        st.header("Generated output:")
        st.text(generated_output)
        st.write("---")
