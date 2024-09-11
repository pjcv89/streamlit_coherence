import streamlit as st
import numpy as np
import pandas as pd
import pickle

from utils_biencoder import search_exact
from utils_biencoder import search_approx
from utils_biencoder import get_cosine_similarity
from utils_biencoder import project_and_plot_from_sentences

from sentence_transformers import SentenceTransformer
import faiss
from safetensors.numpy import load_file


@st.cache_resource()
def load_sentence_transformer():
    model_name = "sentence-transformers/use-cmlm-multilingual"
    sentence_transformer = SentenceTransformer(model_name)
    return sentence_transformer


@st.cache_resource()
def load_categories_embeddings():
    categories_embeddings = pd.read_json(
        "artifacts/biencoder/categories_embeddings.json"
    )
    categories_embeddings["embeddings"] = categories_embeddings["embeddings"].apply(
        np.array
    )
    return categories_embeddings


@st.cache_resource()
def load_linear_layer():
    linear_layer = load_file("artifacts/biencoder/1_Dense/model.safetensors")
    return linear_layer["linear.weight"], linear_layer["linear.bias"]


@st.cache_resource()
def load_projections():
    projections_and_categories = np.load(
        "artifacts/biencoder/projections.npy", allow_pickle=True
    )
    projections_train = projections_and_categories[()]["projections"]
    categories_train = projections_and_categories[()]["categories"]
    del projections_and_categories
    return projections_train, categories_train


@st.cache_resource()
def load_umap_reducer():
    reducer = pickle.load(open("artifacts/biencoder/umap_reducer.bin", "rb"))
    return reducer


@st.cache_resource()
def load_faiss_indexes():
    faiss_index_cats = faiss.read_index(
        "artifacts/biencoder/faiss_index_categories.bin"
    )
    faiss_index_test = faiss.read_index("artifacts/biencoder/faiss_index_test.bin")
    return faiss_index_cats, faiss_index_test


@st.cache_resource()
def load_test_df():
    df = pd.read_csv("data/test.csv", sep="\t")
    return df


def sample_sentence(df, field):
    return df.sample(1)[field].values[0]


def form_callback(sentence1, sentence2, score, feedback):
    encoder_type = "biencoder"
    with open("data/feedback_biencoder.csv", "a+") as f:
        f.write(f"{encoder_type}\t{sentence1}\t{sentence2}\t{score}\t{feedback}\n")


# Main function to run the Streamlit app
def main():

    sentence_transformer = load_sentence_transformer()
    categories_embeddings = load_categories_embeddings()
    weight_matrix, bias_vector = load_linear_layer()
    df = load_test_df()
    threshold = 0.50

    projections_train, categories_train = load_projections()
    reducer = load_umap_reducer()

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = (
            "We regret to inform you that your savings have decreased recently."
        )
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = (
            "Afronta tus proyectos personales con tu préstamo personal."
        )

    st.title("Bi-Encoder")

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
    # SEARCH PARAMETERS #
    search_type = "Exact"
    search_which = "Nearest"
    search_metric = "Euclidean"
    k = 3

    # index = index_cats
    index = None
    input_df = categories_embeddings

    advanced_options = st.sidebar.checkbox(
        "Advanced Options (Will increase memory consumption!)"
    )

    if advanced_options:

        #######################################################
        # LOAD NEEDED ARTIFACTS FOR ADVANCED OPTIONS
        # projections_train, categories_train = load_projections()
        # reducer = load_umap_reducer()
        index_cats, index_test = load_faiss_indexes()
        #######################################################

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
        search_metric = st.sidebar.selectbox("Metric", ("Euclidean", "Inner Product"))
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
    score = np.round(1 - similarity, 4)
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
    ############################################################################
    # STATIC PLOT #
    # if advanced_options:
    if True:
        if st.button("Show Projections of the Two Sentences", key="staticplot"):

            st.markdown(
                """
            How to interpret these plots?

            - By design of the model, incompatible pairs should be far and compatible pairs should be close in the (high dimensional) embedding space.
            - Additionally, after an embedding space is learned by the model, a 2D projected space is built in order to allow visualization and interpretation.
            - The distances between sentences in the projected space seek to reflect the learned compatibility relationships.

            Therefore:
            - Sentence pairs for which the model is very confident to be incoherent (higher scores) will tend to be *far* in the projecteded space.
            - Sentence pairs for which the model is very confident to be coherent (lower scores) will tend to be *close* in the projected space.
            - Sentence pairs for which the model is ambiguous (scores approx around 0.50) will tend to be not too *far* nor too *close* in the projected space.

            Furthermore:
            The projections of the sentences from the training data and their categories are also displayed in softer colors, 
            so we can see in which region each input sentence provided lies within.

            It is recommended to also visualize the interactive plot for the training data (here below) to better understand the regions of the categories
            and examples of sentences in each region. 
            """
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
    st.divider()
    ############################################################################

    ############################################################################
    # FEEDBACK FORM #
    with st.form(key="feedback_form_bi", clear_on_submit=True):

        st.write("**Save current record and provide feedback**")

        feedback = st.text_input("Enter your comments", key="comments")

        submitted = st.form_submit_button("Save")

        if submitted:
            form_callback(
                st.session_state["sentence1"],
                st.session_state["sentence2"],
                score,
                feedback,
            )

    st.info(" #### Current content of the CSV file :point_down:")
    st.dataframe(
        pd.read_csv(
            "data/feedback_biencoder.csv",
            sep="\t",
            names=["encoder", "sentence1", "sentence2", "score", "comments"],
        ),
        height=300,
    )
    st.warning(
        "You can click on the **Download as CSV** option on the top-right corner of the table",
        icon="💾",
    )
    ############################################################################


if __name__ == "__main__":
    main()
