import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt


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


def search_exact(
    sentence_transformer,
    weight_matrix,
    bias_vector,
    input_df,
    sentence,
    mode,
    metric,
    k,
    dummy=None,
):

    embedding = embed(
        sentence_transformer=sentence_transformer,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
        sentence=sentence,
    )

    if mode == "Farthest":
        embedding = (-1) * embedding

    metric_fn = cosine_similarity if metric == "Inner Product" else euclidean_distances
    ascending = False if metric == "Inner Product" else True

    out = input_df.copy()
    out["score"] = out["embeddings"].apply(
        lambda x: metric_fn(x.reshape(1, -1), embedding.reshape(1, -1))[0][0]
    )

    out = (
        out.sort_values(by="score", ascending=ascending)
        .drop(columns=["embeddings"])
        .rename(columns={"category_anchor": "category"})
        .head(k)
    )

    return out


def get_cosine_similarity(
    sentence_transformer,
    weight_matrix,
    bias_vector,
    sentence1,
    sentence2,
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


def search_approx(
    sentence_transformer,
    weight_matrix,
    bias_vector,
    input_df,
    sentence,
    mode,
    metric,
    k,
    index,
):

    if input_df.shape[0] <= 10:
        reference_df = input_df.copy().rename(columns={"category_anchor": "category"})
        cols = ["category"]
    else:
        reference_df = input_df.copy()
        cols = ["category", "text"]

    query = embed(sentence_transformer, weight_matrix, bias_vector, sentence)
    query = np.expand_dims(query, 0)

    if mode == "Farthest":
        query = (-1) * query

    if metric == "Inner Product":
        index.metric_type = 0
    else:
        index.metric_type = 1

    distances, retrieved_indexes = index.search(query, k)
    result = reference_df.iloc[retrieved_indexes[0]][cols]
    result["score"] = np.abs(distances[0].copy())
    return result


def project_and_plot_from_sentences(
    sentence_transformer,
    weight_matrix,
    bias_vector,
    reducer,
    projections_train,
    categories_train,
    sentence1,
    sentence2,
):

    pdf = pd.DataFrame.from_dict({"message": [sentence1, sentence2]})

    array = np.stack(
        pdf["message"]
        .apply(
            lambda sentence: embed(
                sentence_transformer=sentence_transformer,
                weight_matrix=weight_matrix,
                bias_vector=bias_vector,
                sentence=sentence,
            )
        )
        .values
    )

    projections = reducer.transform(array)

    markers = {sentence1: "s", sentence2: "X"}
    hue_order = categories_train.unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        x=projections_train[:, 0],
        y=projections_train[:, 1],
        hue=categories_train,
        hue_order=hue_order,
        alpha=0.25,
    )
    sns.scatterplot(
        x=projections[:, 0],
        y=projections[:, 1],
        hue=pdf["message"],
        style=pdf["message"],
        markers=markers,
        palette="dark",
        s=200,
        ax=ax,
    )
    sns.move_legend(ax, "lower center", bbox_to_anchor=(1, 1))
    plt.xlim(-6, 20)
    plt.ylim(-6, 14)

    return fig
