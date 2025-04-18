from .WgssSimilarity import wgss
import re
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from math import ceil
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def rank_using_tfidf(index_and_sentences_in_cluster):
    sentences = [s for _, s in index_and_sentences_in_cluster]
    tfidf_vectorizer = TfidfVectorizer()
    try:
        # Fit the vectorizer to the Bengali sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    except ValueError:
        return 0
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # Create a dictionary to store the TF-IDF scores for each word in each sentence
    tfidf_scores = {}

    for i in range(len(sentences)):
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        original_index, _ = index_and_sentences_in_cluster[i]
        tfidf_scores[original_index] = {feature_names[index]: tfidf_matrix[i, index] for index in feature_index}

    # Sort sentences based on their average TF-IDF scores
    sorted_sentences = sorted(tfidf_scores.items(), key=lambda x: sum(x[1].values()), reverse=True)

    return sorted_sentences[0][0]


def get_summary(input_text, sigma=0.70710678, proportion=0.15):
    sentence_dividers = ['।', '|', '!', '?', ":", '.', ';']
    divider_pattern = '|'.join(map(re.escape, sentence_dividers))
    tokenizer = RegexpTokenizer(f'[^{divider_pattern}]+|[{divider_pattern}]')
    tokens = tokenizer.tokenize(input_text)

    sentences = []
    for token in tokens:
        if token.strip() and token not in sentence_dividers:
            sentences.append(token.strip())
    
    # initiating affinity matrix
    affinity_matrix = []
    total_sentences = len(sentences)
    for i in range(total_sentences):
        row = [0] * total_sentences
        affinity_matrix.append(row)  # initiating matrix with zeroes
    
    # sentence similarity calculation
    for i in range(total_sentences):
        for j in range(i + 1, total_sentences):
            affinity_matrix[i][j] = wgss(sentences[i], sentences[j], sigma)
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
            sentence_in_this_cluster.append((sentence_index, sentences[sentence_index]))
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
