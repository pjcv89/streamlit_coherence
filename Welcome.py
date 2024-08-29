import streamlit as st
from pages.biencoder import main as main_biencoder
from pages.crossencoder import main as main_crossencoder


def intro():
    import streamlit as st

    st.write("# Welcome to the ISAAC 🍎 Demo! 👋")
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        This is an interactive app to show a prototype of the Interpretable System for Automatic Analysis of Coherence (ISAAC). 
        
        **👈 Select an option from the dropdown on the left side.**

        ### Options:

        - **biencoder**: It uses a Bi-Encoder type of model, which is the one intended to deploy to production, to compute compatibility (i.e. coherence) 
        scores between a pairs of sentences. It provides interpretation in terms of geometry (i.e. embeddings of the sentences and their distances).
        - **crossencoder**: It uses a Cross-Encoder type of model, trained to mirror the behaviour of the Bi-Encoder model through a mechanism called *distillation*, 
        and it's intended to be used in a separate process to extend interpretability capabilities for the user. It provide interpretation in terms of magnitude and direction of word importances.
        - **shared**: It uses both models to compute the compatibility scores, allowing a quick comparison between their outputs.

        ### Recommended reading:
        - [Bi-Encoder vs. Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
        - [Sentence Embeddings: Cross-encoders and Re-ranking. ](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)

    """
    )


if __name__ == "__main__":
    intro()
