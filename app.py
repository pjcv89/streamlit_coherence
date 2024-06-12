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
def load_faiss_indexes():
    faiss_index_cats = faiss.read_index("artifacts/faiss_index_categories_ip.bin")
    faiss_index_test = faiss.read_index("artifacts/faiss_index_test_ip.bin")
    return faiss_index_cats, faiss_index_test


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
    df = load_test_df()
    threshold = 0.80

    # projections_train, categories_train = load_projections()
    # reducer = load_umap_reducer()
    # index_cats, index_test = load_faiss_indexes()

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = "consider diversifying your equity exposure"
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = "your loan is now past due"

    st.title("CDI Coherence - Automatic Labeling Demo")

    ############################################################################
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Sample sentence 1 (English)", key="button1eng"):
            st.session_state["sentence1"] = sample_sentence(df, "text")
    with col2:
        if st.button("Sample sentence 1 (Spanish)", key="button1spa"):
            st.session_state["sentence1"] = sample_sentence(df, "text_esp")
    st.text_area("Sentence 1", st.session_state["sentence1"], key="sentence1")

    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("Sample sentence 2 (English)", key="button2eng"):
            st.session_state["sentence2"] = sample_sentence(df, "text")
    with col4:
        if st.button("Sample sentence 2 (Spanish)", key="button2spa"):
            st.session_state["sentence2"] = sample_sentence(df, "text_esp")
    st.text_area("Sentence 2", st.session_state["sentence2"], key="sentence2")
    ############################################################################
    # SEARCH PARAMETERS #
    search_type = "Exact"
    search_which = "Nearest"
    search_metric = "Inner Product"
    k = 3

    # index = index_cats
    index = None
    input_df = categories_embeddings
    show_static = False

    advanced_options = st.sidebar.checkbox("Advanced Options (Will increase memory!)")

    if advanced_options:

        #######################################################
        # LOAD NEEDED ARTIFACTS FOR ADVANCED OPTIONS
        projections_train, categories_train = load_projections()
        reducer = load_umap_reducer()
        index_cats, index_test = load_faiss_indexes()
        #######################################################

        # show_static = st.checkbox("Show Projections of the Two Sentences")

        which_index = st.sidebar.selectbox(
            "Categories or Test Samples", ("Categories", "Test Samples")
        )

        if which_index == "Categories":
            search_type = st.sidebar.selectbox(
                "Type of Search", ("Exact", "Approximate")
            )
            index = index_cats
            input_df = categories_embeddings
            k_default = 3
            k_max = 10
        else:
            search_type = "Approximate"
            index = index_test
            input_df = df
            k_default = 5
            k_max = 15

        search_which = st.sidebar.selectbox(
            "Nearest or Farthest", ("Nearest", "Farthest")
        )
        search_metric = st.sidebar.selectbox("Metric", ("Inner Product", "Euclidean"))
        k = st.sidebar.slider("Number of outputs in rankings", 1, k_max, k_default)

    ranking_fn = search_exact if search_type == "Exact" else search_approx
    ############################################################################
    st.divider()

    st.write("**Sentence 1:**")
    st.write(st.session_state["sentence1"])
    output1 = ranking_fn(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        input_df,
        st.session_state["sentence1"],
        search_which,
        search_metric,
        k,
        index,
    )

    st.write("**Ranking for Sentence 1:**")
    st.write(output1)

    st.write("**Sentence 2:**")
    st.write(st.session_state["sentence2"])
    output2 = ranking_fn(
        sentence_transformer,
        weight_matrix,
        bias_vector,
        input_df,
        st.session_state["sentence2"],
        search_which,
        search_metric,
        k,
        index,
    )

    st.write("**Ranking for Sentence 2:**")
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
    ############################################################################
    # STATIC PLOT #
    # show_static = st.checkbox("Show Static Plot")
    if advanced_options:
        if st.button("Show Projections of the Two Sentences", key="staticplot"):
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
