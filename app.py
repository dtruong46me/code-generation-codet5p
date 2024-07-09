
import streamlit as st

# Set page title name:
st.set_page_config(page_title="Python Code Generation", page_icon=":robot_face:")

import time
import pandas as pd
import torch


from transformers import TextStreamer, GenerationConfig, AutoTokenizer, T5ForConditionalGeneration

class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, st_container, st_info_container, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.st_container = st_container
        self.st_info_container = st_info_container
        self.text = ""
        self.start_time = None
        self.first_token_time = None
        self.total_tokens = 0

    def on_finalized_text(self, text: str, stream_end: bool=False):
        if self.start_time is None:
            self.start_time = time.time()

        if self.first_token_time is None and len(text.strip()) > 0:
            self.first_token_time = time.time()

        self.text += text

        self.total_tokens += len(text.split())
        self.st_container.markdown("```" + self.text)
        time.sleep(0.03)

        if stream_end:
            total_time = time.time() - self.start_time
            first_token_wait_time = self.first_token_time - self.start_time if self.first_token_time else None
            tokens_per_second = self.total_tokens / total_time if total_time > 0 else None
            
            df = pd.DataFrame(data={
                "First token": [first_token_wait_time],
                "Total tokens": [self.total_tokens],
                "Time taken": [total_time],
                "Token per second": [tokens_per_second]
            })

            self.st_info_container.table(df.T)


def generate_code(model, input_text, generation_config, tokenizer, st_container, st_info_container) -> str:
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Initialize the Streamlit container and streamer
        streamer = StreamlitTextStreamer(tokenizer, st_container, st_info_container, skip_special_tokens=True, decoder_start_token_id=3)

        model.generate(input_ids, streamer=streamer, do_sample=True, generation_config=generation_config)

    except Exception as e:
        raise e


st.title("Python Code Generation")
st.caption("Project 2 - Thầy Tống Văn Vạn")

st.write("---")
input_text = st.text_input("Enter your input")

with st.sidebar:
    st.header("Model:")
    checkpoint = st.selectbox("Model", options=["Choose model", "Salesforce/codet5p-220m-py", "Salesforce/codet5p-770m-py", "dtruong46me/codet5p-220m"])

    st.header("Generation Config")
    temperature = st.number_input("temperature", min_value=0.0, max_value=1.00, step=0.05, value=0.9)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=40)
    top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, step=0.05, value=0.9)
    max_new_tokens = st.number_input("max_new_tokens", min_value=1, max_value=256, value=64)
    min_new_tokens = st.number_input("min_new_tokens", min_value=1, max_value=64, value=4)

generation_config = GenerationConfig(
    min_new_tokens=min_new_tokens,
    max_new_tokens=320,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if checkpoint=="Choose model":
    tokenizer = None
    model = None

if checkpoint!="Choose model":
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

if st.button("Submit"):
    st.write("---")
    st.write("## Code")

    if checkpoint=="Choose model":
        st.error("Please selece a model!")

    else:
        if input_text=="":
            st.error("Please enter a dialogue!")

        st_container = st.empty()
        st_info_container = st.empty()
        generate_code(model, " ".join(input_text.split()), generation_config, tokenizer, st_container, st_info_container)
