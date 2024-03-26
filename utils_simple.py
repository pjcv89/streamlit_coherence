import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def text_cleaning(x):

    x_clean = re.sub("\s+\n+", " ", x)
    x_clean = re.sub("[^a-zA-Z0-9]", " ", x_clean)
    x_clean = x_clean.lower()

    return x_clean


def embed(sentence_transformer, weight_matrix, bias_vector, sentence):
    sentence = text_cleaning(sentence)
    output_transformer = sentence_transformer.encode(sentence)
    embedding = np.maximum(np.dot(weight_matrix, output_transformer) + bias_vector, 0)
    return embedding


def score_sentence(
    sentence_transformer, weight_matrix, bias_vector, categories_embeddings, sentence
):

    embedding = embed(
        sentence_transformer=sentence_transformer,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
        sentence=sentence,
    )

    out = categories_embeddings.copy()
    out["score"] = out["embeddings"].apply(
        lambda x: cosine_similarity(x.reshape(1, -1), embedding.reshape(1, -1))[0][0]
    )

    out = (
        out.sort_values(by="score", ascending=False)
        .drop(columns=["embeddings"])
        .rename(columns={"category_anchor": "category"})
    )

    return out


def get_cosine_similarity(
    sentence_transformer, weight_matrix, bias_vector, sentence1, sentence2
):

    embeddings1 = embed(
        sentence_transformer=sentence_transformer,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
        sentence=sentence1,
    )
    embeddings1 = embeddings1.reshape(1, -1)

    embeddings2 = embed(
        sentence_transformer=sentence_transformer,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
        sentence=sentence2,
    )
    embeddings2 = embeddings2.reshape(1, -1)

    out = np.round(cosine_similarity(embeddings1, embeddings2), 4)[0][0]

    return out
