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