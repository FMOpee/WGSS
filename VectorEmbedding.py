from functools import lru_cache
import fasttext.util


@lru_cache(maxsize=1)
def __model_loader():
    print ("    2.1. loading fasttext model")
    fasttext.util.download_model('bn', if_exists='ignore')
    ft = fasttext.load_model('cc.bn.300.bin')
    return ft


def vectorizer(sentence):
    model = __model_loader()

    vectors_of_words_in_sentence = []
    for word in sentence:
        vector = model.get_word_vector(word)
        vectors_of_words_in_sentence.append(vector)

    return vectors_of_words_in_sentence
