import streamlit as st
import numpy as np
import pandas as pd

from utils_crossencoder import get_prediction_from_strings, get_shap_values_from_strings
from utils_crossencoder import plot_attribution

from transformers_interpret import PairwiseSequenceClassificationExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import shap

from streamlit_shap import st_shap
import torch


@st.cache_resource()
def load_cross_encoder_artifacts():
    model_path = "artifacts/crossencoder"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        # return_all_scores=True,
        top_k=1,
        device=torch.device("cpu"),
    )
    # explainer = shap.Explainer(pipe)
    explainer_shap = shap.Explainer(
        shap.models.TransformersPipeline(pipe, rescale_to_logits=True)
    )
    explainer_ti = PairwiseSequenceClassificationExplainer(model, tokenizer)
    return tokenizer, pipe, explainer_shap, explainer_ti


@st.cache_resource()
def load_test_df():
    df = pd.read_csv("data/test.csv", sep="\t")
    return df


def sample_sentence(df, field):
    return df.sample(1)[field].values[0]


# Main function to run the Streamlit app
def main():

    tokenizer, pipe, explainer_shap, explainer_ti = load_cross_encoder_artifacts()
    df = load_test_df()
    threshold = 0.60

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = (
            "We regret to inform you that your savings have decreased recently."
        )
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = (
            "Afronta tus proyectos personales con tu préstamo personal."
        )

    st.title("Cross-Encoder")

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

    st.write("**Sentence 1:**")
    st.write(st.session_state["sentence1"])

    st.write("**Sentence 2:**")
    st.write(st.session_state["sentence2"])

    st.divider()
    prediction = get_prediction_from_strings(
        pipe=pipe,
        text_x=st.session_state["sentence1"],
        text_y=st.session_state["sentence2"],
    )
    score = np.round(prediction[0][0]["score"], 4)
    st.write("**Incompability score (higher means more incompatible)**")
    st.write(score)

    if score <= threshold:
        st.warning(
            "Messages seem compatible (at threshold: " + str(threshold) + ")",
            icon="✅",
        )
    else:
        st.warning(
            "Messages seem non-compatible (at threshold: " + str(threshold) + ")",
            icon="⚠️",
        )
    st.divider()

    if st.button("Compute Explanations", key="explainers"):

        st.write("**Transformers-Interpret**")

        st.markdown(
            """
        Barplots: How to interpret these plots?

        Here, the size of the bars indicate the magnitud of the influence of each word.
        Regarding direction, words that have bars pointing to the right side are pushing the score towards higher values
        while words that have bars pointing to the left side are pushing the score toward lower values.

        It is important to note that in this case of sentence pairs classification, the magnitud and direction of each word
        is dependant on the presence of the words in the same sentence (*self-attention* mechanism) and the presence of words 
        in the other sentence with which is interacting (*cross-attention* mechanism).

        Therefore:
        - Sentence pairs for which the model is very confident to be incoherent (higher scores) will tend to more words pointing to the right in both sentences.
        - Sentence pairs for which the model is very confident to be coherent (lower scores) will tend to have more words pointing to the left in both sentences.
        - Sentence pairs for which the model is ambiguous (scores approx around 0.50) will tend to have words pointing to both sides in both sentences.
        """
        )

        fig_1, fig_2, fig_matrix = plot_attribution(
            explainer=explainer_ti,
            tokenizer=tokenizer,
            text_x=st.session_state["sentence1"],
            text_y=st.session_state["sentence2"],
        )

        st.pyplot(fig_1)
        st.pyplot(fig_2)

        st.markdown(
            """
        Heatmap: How to interpret these plots?

        Here, the interactions between words of both sentences are shown.
        - Pairs of words that influence the model prediction to be incoherent will have entries with higher (clearer) values.
        - Pairs of words that influence the model prediction to be cooherent will have entries with lower (darker) values.
        """
        )
        st.pyplot(fig_matrix)

        st.divider()
        st.write("**SHAP**")

        st.markdown(
            """
        Shap plots: How to interpret these plots?

        Here, red regions correspond to parts of the text that increase the output of the model when they are included,
        while blue regions decrease the output of the model when they are included. 
          
        In the context of this use case, here red corresponds to more incompatible (incoherent) and blue to more compatible (coherent).

        It is important to note that in this case of sentence pairs classification, the color of words and group of words in one sentence
        is dependant on the presence of the words in the same sentence (*self-attention* mechanism) and the presence of words 
        in the other sentence with which is interacting (*cross-attention* mechanism).

        Therefore:
        - Sentence pairs for which the model is very confident to be incoherent (higher scores) will tend to have red regions in both sentences.
        - Sentence pairs for which the model is very confident to be coherent (lower scores) will tend to have blue regions in both sentences.
        - Sentence pairs for which the model is ambiguous (scores approx around 0.50) will tend to be both red and blue regions.
        """
        )
        shap_values = get_shap_values_from_strings(
            explainer=explainer_shap,
            text_x=st.session_state["sentence1"],
            text_y=st.session_state["sentence2"],
        )

        st_shap(shap.plots.text(shap_values), height=300)

        st.divider()


if __name__ == "__main__":
    main()
