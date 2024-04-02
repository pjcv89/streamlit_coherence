import streamlit as st
import numpy as np
import pandas as pd

# import torch
# from utils import score_sentence, get_cosine_similarity
# from models import UniversalSentenceEncoder, SiameseModel
from utils_simple import score_sentence, get_cosine_similarity
from sentence_transformers import SentenceTransformer


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
def load_model():
    device = (
        torch.device("mps")
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        else torch.device("cpu")
    )
    model_name = "sentence-transformers/use-cmlm-multilingual"
    PATH = "artifacts/siamese_network_use.pth"

    shared_encoder = UniversalSentenceEncoder(model_name)
    shared_encoder.to(device)
    siamese_model = SiameseModel(shared_encoder)
    siamese_model.load_state_dict(torch.load(PATH))
    return siamese_model


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

    if "sentence1" not in st.session_state:
        st.session_state["sentence1"] = "your investment has increased"
    if "sentence2" not in st.session_state:
        st.session_state["sentence2"] = "you should make timely payments"

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
        st.write("Sentence 1:")
        st.write(st.session_state["sentence1"])
        # output1 = score_sentence(
        #    siamese_model, categories_embeddings, st.session_state["sentence1"]
        # )
        output1 = score_sentence(
            sentence_transformer,
            weight_matrix,
            bias_vector,
            categories_embeddings,
            st.session_state["sentence1"],
        )
        st.write("Ranking of categories for Sentence 1:")
        st.write(output1)

        if st.button("Sample sentence 2", key="button2"):
            st.session_state["sentence2"] = sample_sentence(df, field)
        st.write("Sentence 2:")
        st.write(st.session_state["sentence2"])
        # output2 = score_sentence(
        #    siamese_model, categories_embeddings, st.session_state["sentence2"]
        # )
        output2 = score_sentence(
            sentence_transformer,
            weight_matrix,
            bias_vector,
            categories_embeddings,
            st.session_state["sentence2"],
        )
        st.write("Ranking of categories for Sentence 2:")
        st.write(output2)

        # similarity = get_cosine_similarity(
        #    siamese_model, st.session_state["sentence1"], st.session_state["sentence2"]
        # )
        similarity = get_cosine_similarity(
            sentence_transformer,
            weight_matrix,
            bias_vector,
            st.session_state["sentence1"],
            st.session_state["sentence2"],
        )
        st.write("Compability score (higher means more compatible)")
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
    else:
        sentence1 = st.text_input("Sentence 1", value="your investment has increased")
        if sentence1 != "":
            # output1 = score_sentence(siamese_model, categories_embeddings, sentence1)
            output1 = score_sentence(
                sentence_transformer,
                weight_matrix,
                bias_vector,
                categories_embeddings,
                sentence1,
            )
            st.write("Ranking of categories for Sentence 1:")
            st.write(output1)

        sentence2 = st.text_input("Sentence 2", value="you should make timely payments")
        if sentence2 != "":
            # output2 = score_sentence(siamese_model, categories_embeddings, sentence2)
            output2 = score_sentence(
                sentence_transformer,
                weight_matrix,
                bias_vector,
                categories_embeddings,
                sentence2,
            )
            st.write("Ranking of categories for Sentence 2:")
            st.write(output2)

        # Button to process both texts
        if sentence1 and sentence2:
            # similarity = get_cosine_similarity(siamese_model, sentence1, sentence2)
            similarity = get_cosine_similarity(
                sentence_transformer, weight_matrix, bias_vector, sentence1, sentence2
            )
            st.write("Compability score (higher means more compatible)")
            st.write(similarity)
            if similarity >= threshold:
                st.warning(
                    "Messages seem compatible (at threshold: " + str(threshold) + ")",
                    icon="✅",
                )
            else:
                st.warning(
                    "Messages seem non-compatible (at threshold: "
                    + str(threshold)
                    + ")",
                    icon="⚠️",
                )


if __name__ == "__main__":
    if check_password():
        main()
