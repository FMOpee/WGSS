import re
from nltk.tokenize import RegexpTokenizer
import pkg_resources


def __custom_tokenizer(text, dividers):
    # idk what this regex part is doing. got it from stack overflow.
    divider_pattern = '|'.join(map(re.escape, dividers))
    tokenizer = RegexpTokenizer(f'[^{divider_pattern}]+|[{divider_pattern}]')
    tokens = tokenizer.tokenize(text)

    # removes the divider tokens and also strips them
    processed_tokens = []
    for token in tokens:
        if token.strip() and token not in dividers:
            processed_tokens.append(token.strip())

    return processed_tokens


def __stopword_removal(sentences):
    # Use pkg_resources to get the path to stopwords_bn.txt
    stopwords_path = pkg_resources.resource_filename(__name__, "stopwords_bn.txt")
    # Read the stopwords
    with open(stopwords_path, "r", encoding="utf8") as f:
        stop_words = f.readlines()
    # Process and return the stopwords
    stop_words = [stop_word.strip() for stop_word in stop_words]

    preprocessed_word_list = []
    for sentence in sentences:
        preprocessed_word_list.append([word for word in sentence if word not in stop_words])

    return preprocessed_word_list


def preprocessor(text):
    sentence_dividers = ['।', '|', '!', '?', ":", '.', ';']
    sentences = __custom_tokenizer(text, sentence_dividers)  # each sentences as intact and in a list

    word_dividers = [' ', ',', '.', ';', '"', "'", '`', '(', ')', '[', ']', '-', '‘', '’‌', '%', '/', '\\']
    words = []
    for sentence in sentences:
        words.append(__custom_tokenizer(sentence, word_dividers))  # each word as a separate token now

    preprocessed_word_list = __stopword_removal(words)

    # the returning variables are two list.
    # sentences = [sent1, sent2, sent3, ...]
    # preprocessed_word_list = [[word11, word12, ...], [word21, word22, ...], ...]
    return sentences, preprocessed_word_list
