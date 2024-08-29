import streamlit as st
import numpy as np
import pandas as pd

from utils_biencoder import get_cosine_similarity
from utils_crossencoder import get_prediction_from_strings

from sentence_transformers import SentenceTransformer
from safetensors.numpy import load_file

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import torch


@st.cache_resource()
def load_sentence_transformer():
    model_name = "sentence-transformers/use-cmlm-multilingual"
    sentence_transformer = SentenceTransformer(model_name)
    return sentence_transformer


@st.cache_resource()
def load_linear_layer():
    linear_layer = load_file("artifacts/biencoder/1_Dense/model.safetensors")
    return linear_layer["linear.weight"], linear_layer["linear.bias"]


@st.cache_resource()
def load_cross_encoder_artifacts():
    model_path = "artifacts/crossencoder"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=torch.device("cpu"),
    )
    return pipe


@st.cache_resource()
def load_test_df():
    df = pd.read_csv("data/test.csv", sep="\t")
    return df


def sample_sentence(df, field):
    return df.sample(1)[field].values[0]


# Main function to run the Streamlit app
def main():

    sentence_transformer = load_sentence_transformer()
    weight_matrix, bias_vector = load_linear_layer()

    pipe = load_cross_encoder_artifacts()

    df = load_test_df()
    threshold_bi = 0.50
    threshold_cross = 0.60

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = (
            "We regret to inform you that your savings have decreased recently."
        )
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = (
            "Afronta tus proyectos personales con tu préstamo personal."
        )

    st.title("Comparing Bi-Encoder and Cross-Encoder outputs")

    ############################################################################
    col1, _ = st.columns([1, 1])
    with col1:
        if st.button("Sample sentence 1", key="button1eng"):
            st.session_state["sentence1"] = sample_sentence(df, "text")
    # with col2:
    #     if st.button("Sample sentence 1 (Spanish)", key="button1spa"):
    #         st.session_state["sentence1"] = sample_sentence(df, "text_esp")
    st.text_area("Sentence 1", st.session_state["sentence1"], key="sentence1")

    col3, _ = st.columns([1, 1])
    with col3:
        if st.button("Sample sentence 2", key="button2eng"):
            st.session_state["sentence2"] = sample_sentence(df, "text")
    # with col4:
    #     if st.button("Sample sentence 2 (Spanish)", key="button2spa"):
    #         st.session_state["sentence2"] = sample_sentence(df, "text_esp")
    st.text_area("Sentence 2", st.session_state["sentence2"], key="sentence2")
    ############################################################################
    ############################################################################
    st.divider()

    similarity = get_cosine_similarity(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        st.session_state["sentence1"],
        st.session_state["sentence2"],
    )
    score = np.round(1 - similarity, 4)
    st.write("**BI-ENCODER**")
    st.write("**Incompability score (higher means more incompatible)**")
    st.write(score)

    if score <= threshold_bi:
        st.warning(
            "Messages seem compatible (at threshold: " + str(threshold_bi) + ")",
            icon="✅",
        )
    else:
        st.warning(
            "Messages seem non-compatible (at threshold: " + str(threshold_bi) + ")",
            icon="⚠️",
        )

    st.divider()
    prediction = get_prediction_from_strings(
        pipe=pipe,
        text_x=st.session_state["sentence1"],
        text_y=st.session_state["sentence2"],
    )
    score = np.round(prediction[0][0]["score"], 4)
    st.write("**CROSS-ENCODER**")
    st.write("**Incompability score (higher means more incompatible)**")
    st.write(score)

    if score <= threshold_cross:
        st.warning(
            "Messages seem compatible (at threshold: " + str(threshold_cross) + ")",
            icon="✅",
        )
    else:
        st.warning(
            "Messages seem non-compatible (at threshold: " + str(threshold_cross) + ")",
            icon="⚠️",
        )
    st.divider()


if __name__ == "__main__":
    main()
