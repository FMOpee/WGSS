from collections import defaultdict
from math import ceil
import json
from sklearn.cluster import SpectralClustering
import os
import pkg_resources

from .Preprocessor import preprocessor
from .VectorEmbedding import vectorizer
from .WGSS import sentence_similarity as sentence_similarity_calculator
from .TFIDF import rank_using_tfidf


def preprocessing(input_document):
    return preprocessor(input_document)


def vectorize(word_array):
    return vectorizer(word_array)


def sentence_similarity(vector_array_1, vector_array_2, sigma=5e-11):
    return sentence_similarity_calculator(vector_array_1, vector_array_2, sigma)


def get_bengali_stop_words():
    # Use pkg_resources to get the path to stopwords_bn.txt
    stopwords_path = pkg_resources.resource_filename(__name__, "stopwords_bn.txt")
    # Read the stopwords
    with open(stopwords_path, "r", encoding="utf8") as f:
        stop_words = f.readlines()
    # Process and return the stopwords
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def get_curated_summarization_dataset():
    return json.load(open("self_curated_dataset.json", "r", encoding="utf8"))


def get_Bengali_summary(input_document, sigma=5e-11, proportion=0.2):
    # pre_processing
    sentences, word_arrays = preprocessing(input_document)

    # word embedding
    set_of_vectors = []
    for word_array in word_arrays:
        set_of_vectors.append(vectorize(word_array))

    # initiating affinity matrix
    affinity_matrix = []
    total_sentences = len(set_of_vectors)
    for i in range(total_sentences):
        row = [0] * total_sentences
        affinity_matrix.append(row)  # initiating matrix with zeroes

    # sentence similarity calculation
    for i in range(total_sentences):
        for j in range(i + 1, total_sentences):
            affinity_matrix[i][j] = sentence_similarity(set_of_vectors[i], set_of_vectors[j], sigma)
            affinity_matrix[j][i] = affinity_matrix[i][j]

    # setting number of clusters
    n_clusters = max(min(2, total_sentences),
                     ceil(total_sentences * proportion))

    # clustering
    if len(affinity_matrix) > 1:
        model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        model.fit(affinity_matrix)

        # a dictionary containing the labels and a list of sentence index in that cluster is produced
        set_of_clusters = defaultdict(list)
        for idx, label in enumerate(model.labels_):
            set_of_clusters[label].append(idx)  # cluster_indices = { "1" : [0,2,4...], "2":[1,3,5...],...}
    else:
        set_of_clusters = {"1": [0]}

    # picking the best sentence from each cluster
    indices_in_summary = []
    for cluster in set_of_clusters.items():
        sentence_in_this_cluster = []
        for sentence_index in cluster[1]:
            sentence_in_this_cluster.append(sentences[sentence_index])
        if len(sentence_in_this_cluster) > 1:
            picked_index = rank_using_tfidf(sentence_in_this_cluster)
        else:
            picked_index = cluster[1][0]
        indices_in_summary.append(picked_index)

    # sorting indices in their order of appearance
    for i in range(len(indices_in_summary)):
        for j in range(i + 1, len(indices_in_summary)):
            if indices_in_summary[i] > indices_in_summary[j]:
                indices_in_summary[i], indices_in_summary[j] = indices_in_summary[j], indices_in_summary[i]

    # summary generation
    summary = ''
    for index in indices_in_summary:
        summary += sentences[index] + '। '

    return summary
