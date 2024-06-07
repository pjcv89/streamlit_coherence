import streamlit as st
import numpy as np
import pandas as pd
import pickle

# import torch
# from models import UniversalSentenceEncoder, SiameseModel
from utils import search_exact
from utils import search_approx
from utils import get_cosine_similarity
from utils import project_and_plot_from_sentences

from sentence_transformers import SentenceTransformer
import faiss


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


@st.cache_resource()
def load_sentence_transformer():
    model_name = "sentence-transformers/use-cmlm-multilingual"
    sentence_transformer = SentenceTransformer(model_name)
    return sentence_transformer


@st.cache_resource()
def load_categories_embeddings():
    categories_embeddings = pd.read_json("artifacts/categories_embeddings.json")
    categories_embeddings["embeddings"] = categories_embeddings["embeddings"].apply(
        np.array
    )
    return categories_embeddings


@st.cache_resource()
def load_linear_layer():
    linear_layer = np.load("artifacts/linear_layer.npy", allow_pickle=True)
    bias_vector = linear_layer[()]["bias_vector"]
    weight_matrix = linear_layer[()]["weight_matrix"]
    del linear_layer
    return weight_matrix, bias_vector


@st.cache_resource()
def load_projections():
    projections_and_categories = np.load("artifacts/projections.npy", allow_pickle=True)
    projections_train = projections_and_categories[()]["projections"]
    categories_train = projections_and_categories[()]["categories"]
    del projections_and_categories
    return projections_train, categories_train


@st.cache_resource()
def load_umap_reducer():
    reducer = pickle.load(open("artifacts/umap_reducer.bin", "rb"))
    return reducer


@st.cache_resource()
def load_faiss_index():
    faiss_index = faiss.read_index("artifacts/faiss_index_categories_ip.bin")
    return faiss_index


@st.cache_resource()
def load_test_df():
    df = pd.read_csv("data/test.csv")
    return df


def sample_sentence(df, field):
    return df.sample(1)[field].values[0]


# Main function to run the Streamlit app
def main():

    # siamese_model = load_model()
    sentence_transformer = load_sentence_transformer()
    categories_embeddings = load_categories_embeddings()
    weight_matrix, bias_vector = load_linear_layer()
    projections_train, categories_train = load_projections()
    reducer = load_umap_reducer()
    index = load_faiss_index()
    df = load_test_df()
    threshold = 0.80

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = "consider diversifying your equity exposure"
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = "your loan is now past due"

    st.title("CDI Coherence - Automatic Labeling Demo")

    option = st.sidebar.selectbox(
        "Select Option", ("Sampling Sentences", "Enter Inputs")
    )

    if option == "Sampling Sentences":
        option = st.sidebar.selectbox("Select Language", ("English", "Spanish"))
        if option == "English":
            field = "text"
        else:
            field = "text_esp"

        if st.button("Sample sentence 1", key="button1"):
            st.session_state["sentence1"] = sample_sentence(df, field)

        if st.button("Sample sentence 2", key="button2"):
            st.session_state["sentence2"] = sample_sentence(df, field)

    else:
        st.session_state["sentence1"] = st.text_input(
            "Sentence 1", value="consider diversifying your equity exposure"
        )
        st.session_state["sentence2"] = st.text_input(
            "Sentence 2", value="your loan is now past due"
        )
    ############################################################################
    # SEARCH PARAMETERS #
    search_type = "Exact"
    search_which = "Near"
    search_metric = "Inner Product"
    k = 3

    advanced_options = st.sidebar.checkbox("Advanced Options")

    if advanced_options:
        search_type = st.sidebar.selectbox("Type of Search", ("Exact", "Approximate"))
        search_which = st.sidebar.selectbox("Mode", ("Near", "Far"))
        search_metric = st.sidebar.selectbox("Metric", ("Inner Product", "Euclidean"))
        k = st.sidebar.slider("Number of categories", 1, 10, 3)

    ranking_fn = search_exact if search_type == "Exact" else search_approx
    ############################################################################
    st.divider()

    st.write("**Sentence 1:**")
    st.write(st.session_state["sentence1"])
    output1 = ranking_fn(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        categories_embeddings,
        st.session_state["sentence1"],
        search_which,
        search_metric,
        k,
        index,
    )

    st.write("**Ranking of categories for Sentence 1:**")
    st.write(output1)

    st.write("**Sentence 2:**")
    st.write(st.session_state["sentence2"])
    output2 = ranking_fn(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        categories_embeddings,
        st.session_state["sentence2"],
        search_which,
        search_metric,
        k,
        index,
    )

    st.write("**Ranking of categories for Sentence 2:**")
    st.write(output2)

    similarity = get_cosine_similarity(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        st.session_state["sentence1"],
        st.session_state["sentence2"],
    )
    st.write("**Compability score (higher means more compatible)**")
    st.write(similarity)

    if similarity >= threshold:
        st.warning(
            "Messages seem compatible (at threshold: " + str(threshold) + ")",
            icon="✅",
        )
    else:
        st.warning(
            "Messages seem non-compatible (at threshold: " + str(threshold) + ")",
            icon="⚠️",
        )

    st.write("**PROJECTIONS OF THE TWO SENTENCES**")
    fig = project_and_plot_from_sentences(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        reducer,
        projections_train,
        categories_train,
        st.session_state["sentence1"],
        st.session_state["sentence2"],
    )

    st.pyplot(fig)

    st.divider()
    ############################################################################
    # INTERACTIVE PLOTS #
    show_interactive = st.checkbox("Show Interactive Plots")

    if show_interactive:
        st.header("PROJECTIONS OF TRAINING AND TEST DATA")
        which = st.selectbox("Set of messages", ("Training", "Test"))
        if which == "Training":
            file_name = "umap_train.html"
        elif which == "Test":
            file_name = "umap_test.html"

        path_to_html = "visualization/" + file_name
        with open(path_to_html, "r") as f:
            html_data = f.read()
        st.download_button(label="Download HTML", data=html_data, file_name=file_name)
        st.components.v1.html(html_data, width=1000, height=1000, scrolling=False)
    ############################################################################


if __name__ == "__main__":
    if check_password():
        main()
