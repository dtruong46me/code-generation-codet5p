import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig

# Define a list of model options
MODEL_OPTIONS = {
    "CodeT5": "Salesforce/codet5-base",
    "FinetunedT5-LoRA": "Dinosaur1812/codet5p-770m-lora"
}

# Load the models and tokenizers
models = {}
tokenizers = {}

for name, model_name in MODEL_OPTIONS.items():
    if "LoRA" in name:
        config = PeftConfig.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model

def generate_code(model_name, prompt):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=150, num_return_sequences=1)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

# Streamlit UI
st.title("Code Generation with LLM")

# Dropdown for model selection
model_name = st.selectbox("Choose a model:", options=list(MODEL_OPTIONS.keys()))

# Text box for entering the NL prompt
nl_prompt = st.text_area("Enter your natural language prompt:", "Write a function to reverse a list")

if st.button("Generate Code"):
    with st.spinner(f"Generating code using {model_name}..."):
        code = generate_code(MODEL_OPTIONS[model_name], nl_prompt)
        st.code(code, language='python')
