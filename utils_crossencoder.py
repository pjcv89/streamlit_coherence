import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer
from transformers_interpret import PairwiseSequenceClassificationExplainer
from transformers import pipeline
import shap


def get_prediction_from_strings(pipe: pipeline, text_x: str, text_y: str):

    dict_of_text_pairs = {"text": text_x, "text_pair": text_y}

    prediction = pipe([dict_of_text_pairs])

    return prediction


def get_shap_values_from_strings(explainer: shap.Explainer, text_x: str, text_y: str):

    dict_of_text_pairs = {"text": text_x, "text_pair": text_y}

    joined_text = (
        "[CLS] "
        + dict_of_text_pairs["text"]
        + " [SEP] "
        + dict_of_text_pairs["text_pair"]
        + " [SEP] "
    )

    shap_values = explainer([joined_text], fixed_context=None)

    return shap_values


def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    """
    This function sorts the output of the attributes array for plotting purpose
    """
    tup.sort(key=lambda x: x[1])
    return tup


def plot_importances(list_of_tuples, pred_class, top_values=10):
    """
    This function plots the top X values. If incoherent, plots the most positives, if coherent the most negatives (attr. values)
    """
    sorted = Sort_Tuple(list_of_tuples.copy())
    if pred_class == "Incoherent":
        labels, values = zip(*(sorted[-top_values:]))
    else:
        labels, values = zip(*(sorted[:top_values]))

    # labels, values = zip(*(sorted))
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (12, 10)
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(values)), labels)
    # plt.show()
    return fig


def plot_matrix(
    mat_sum,
    tokens_sentence1,
    tokens_sentence2,
    text_x: str,
    text_y: str,
    pred_class_label: float,
):
    # Create subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create matrix heatplot from attributes multiplication
    cax = ax.matshow(mat_sum, interpolation="nearest")
    cax2 = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    fig.colorbar(cax, cax=cax2)

    # Create axis
    xaxis = np.arange(len(tokens_sentence1))
    yaxis = np.arange(len(tokens_sentence2))

    # Set ticks on axis and labels
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xticklabels(tokens_sentence1)
    ax.set_yticklabels(tokens_sentence2)

    ##Add title
    figure_title = "Pairwise attributes effect: " + r"$\bf{" + pred_class_label + "}$"
    plt.text(
        0.5,
        1.15,
        figure_title,
        horizontalalignment="center",
        fontsize=13,
        transform=ax.transAxes,
    )
    u1 = "Msg 1:  "
    u2 = "Msg 2:  "

    ax.text(
        0.5,
        -0.15,
        r"$\bf{" + u1 + "}$" + text_x + "\n\n" + r"$\bf{" + u2 + "}$" + text_y,
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
        wrap=True,
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    # plt.show()
    return fig


def get_from_attr(attributions_tuple, index_sep, tokens=True):
    """
    This function returns attribution values from pairwise array (if tokens=False) and tokenks (if tokens=True -- default)
    """
    if tokens:
        tokens_sentence1 = list(
            np.array([x[0] for x in attributions_tuple[1:index_sep]])
        )
        tokens_sentence2 = list(
            np.array([x[0] for x in attributions_tuple[index_sep + 1 : -1]])
        )
    else:
        tokens_sentence1 = list(
            np.array([x[1] for x in attributions_tuple[1:index_sep]])
        )
        tokens_sentence2 = list(
            np.array([x[1] for x in attributions_tuple[index_sep + 1 : -1]])
        )
    return tokens_sentence1, tokens_sentence2


def get_map_sentence_embedding(tokenizer, text_x, text_y):
    """
    This function returns offset mapping of tokenizer. This is used to identify tokens corresponding to same word
    """
    embs = tokenizer(text_x + tokenizer.sep_token + text_y, return_offsets_mapping=True)
    # embs = tokenizer(sample['text_x'].values[0] + tokenizer.sep_token + sample['text_y'].values[0], return_offsets_mapping=True)
    maps = embs["offset_mapping"]
    return maps


def get_pairwise(toks, attrs, maps, puncts):
    """
    This function return attribution pairiwse format given the toks, values and mapping. it removes punctuation or symbols
    """

    pairwise1 = []
    punct_indices = []
    for list_index, element in enumerate(toks):
        if element in puncts:
            punct_indices.append((list_index, element))

    for x in sorted(punct_indices)[::-1]:
        toks.pop(x[0])  # remove this elements from list. not interested
    for x in sorted(punct_indices)[::-1]:
        attrs.pop(x[0])
    for x in sorted(punct_indices)[::-1]:
        maps.pop(x[0])  # remove this elements from list. not interested

    for i in range(len(toks)):
        pairwise1 = pairwise1 + [(toks[i], attrs[i])]
    return pairwise1


def aggregate_attributions(attributions):
    """
    This function aggregates words of given attribution values given at token-level
    """
    aggregated = defaultdict(float)
    current_word = ""
    current_score = 0.0

    for token, value in attributions:
        if token.startswith("##"):
            # This is a subword, append it to the current word.
            current_word += token[2:]  # Remove the "##" prefix
            current_score += value  # Add value to the current score
        else:
            # If there is a current word, add its score to the dictionary
            if current_word:
                aggregated[current_word] += current_score

            # Start a new current word and reset the current score
            current_word = token
            current_score = value  # Reset score for the new word

    # After the loop, add the last word and its score
    if current_word:
        aggregated[current_word] += current_score

    return dict(aggregated)


def remove_short_words(data_dict, max_length=4):
    """
    This function removes words smaller than max length to avoid useless info to the user
    """
    for k, v in list(data_dict.items()):
        if len(k) < max_length:
            del data_dict[k]
    return data_dict


def plot_attribution(
    explainer: PairwiseSequenceClassificationExplainer,
    tokenizer: AutoTokenizer,
    text_x: str,
    text_y: str,
):
    """
    This function plots the attribution bars given a sample is the format of dataframe where 'text_x' is 'Message1' and 'text_y' is 'Message 2'
    """

    pairwise_attr = explainer(
        text_x,
        text_y,
        flip_sign=False,
        # class_name="score_soft"
    )

    index_sep = [i for i, tupl in enumerate(pairwise_attr) if tupl[0] == "[SEP]"][0]

    map_emb = get_map_sentence_embedding(tokenizer, text_x, text_y)

    maps1 = map_emb[:index_sep]
    maps2 = map_emb[index_sep + 1 :]

    toks1, toks2 = get_from_attr(pairwise_attr, index_sep, tokens=True)
    attr1, attr2 = get_from_attr(pairwise_attr, index_sep, tokens=False)

    puncts = [
        ".",
        "?",
        "¿",
        "!",
        "¿",
        "]",
        ",",
        "%",
        "BBVA",
        "bbva",
        "[",
        "[CLS]",
        "[SEP]",
    ]

    pairwise1 = get_pairwise(toks1, attr1, maps1, puncts)
    pairwise2 = get_pairwise(toks2, attr2, maps2, puncts)

    # Call the function with the example input
    aggregated_attribution_scores1 = aggregate_attributions(pairwise1)
    aggregated_attribution_scores2 = aggregate_attributions(pairwise2)

    #####################
    # Delete short words
    aggregated_attribution_scores1 = remove_short_words(aggregated_attribution_scores1)
    aggregated_attribution_scores2 = remove_short_words(aggregated_attribution_scores2)

    if explainer.pred_probs < 0.60:
        pred_class_label = "Coherent"
    else:
        pred_class_label = "Incoherent"

    fig_1 = plot_importances(
        list(aggregated_attribution_scores1.items()), pred_class_label
    )
    fig_2 = plot_importances(
        list(aggregated_attribution_scores2.items()), pred_class_label
    )

    #############################################################################
    tokens_sentence1 = np.array(list(aggregated_attribution_scores1.keys()))
    tokens_sentence2 = np.array(list(aggregated_attribution_scores2.keys()))

    list_attr_sentence1 = np.array(list(aggregated_attribution_scores1.values()))
    list_attr_sentence2 = np.array(list(aggregated_attribution_scores2.values()))

    mat_sum = np.add.outer(list_attr_sentence2, list_attr_sentence1)
    fig_matrix = plot_matrix(
        mat_sum,
        tokens_sentence1,
        tokens_sentence2,
        text_x,
        text_y,
        pred_class_label,
    )
    #############################################################################

    return fig_1, fig_2, fig_matrix
