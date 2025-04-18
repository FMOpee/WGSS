import numpy
from functools import lru_cache
import fasttext.util
import re
from nltk.tokenize import RegexpTokenizer

stopwords = ["অবশ্য", "অনেক", "অনেকে", "অনেকেই", "অন্তত", "অথবা", "অথচ", "অর্থাত", "অন্য", "আজ", "আছে", "আপনার", "আপনি", "আবার", "আমরা", "আমাকে", "আমাদের", "আমার", "আমি", "আরও", "আর", "আগে", "আগেই", "আই", "অতএব", "আগামী", "অবধি", "অনুযায়ী", "আদ্যভাগে", "এই", "একই", "একে", "একটি", "এখন", "এখনও", "এখানে", "এখানেই", "এটি", "এটা", "এটাই", "এতটাই", "এবং", "একবার", "এবার", "এদের", "এঁদের", "এমন", "এমনকী", "এল", "এর", "এরা", "এঁরা", "এস", "এত", "এতে", "এসে", "একে", "এ", "ঐ", "ই", "ইহা", "ইত্যাদি", "উনি", "উপর", "উপরে", "উচিত", "ও", "ওই", "ওর", "ওরা", "ওঁর", "ওঁরা", "ওকে", "ওদের", "ওঁদের", "ওখানে", "কত", "কবে", "করতে", "কয়েক", "কয়েকটি", "করবে", "করলেন", "করার", "কারও", "করা", "করি", "করিয়ে", "করার", "করাই", "করলে", "করলেন", "করিতে", "করিয়া", "করেছিলেন", "করছে", "করছেন", "করেছেন", "করেছে", "করেন", "করবেন", "করায়", "করে", "করেই", "কাছ", "কাছে", "কাজে", "কারণ", "কিছু", "কিছুই", "কিন্তু", "কিংবা", "কি", "কী", "কেউ", "কেউই", "কাউকে", "কেন", "কে", "কোনও", "কোনো", "কোন", "কখনও", "ক্ষেত্রে", "খুব", "গুলি", "গিয়ে", "গিয়েছে", "গেছে", "গেল", "গেলে", "গোটা", "চলে", "ছাড়া", "ছাড়াও", "ছিলেন", "ছিল", "জন্য", "জানা", "ঠিক", "তিনি", "তিনঐ", "তিনিও", "তখন", "তবে", "তবু", "তাঁদের", "তাঁাহারা", "তাঁরা", "তাঁর", "তাঁকে", "তাই", "তেমন", "তাকে", "তাহা", "তাহাতে", "তাহার", "তাদের", "তারপর", "তারা", "তারৈ", "তার", "তাহলে", "তিনি", "তা", "তাও", "তাতে", "তো", "তত", "তুমি", "তোমার", "তথা", "থাকে", "থাকা", "থাকায়", "থেকে", "থেকেও", "থাকবে", "থাকেন", "থাকবেন", "থেকেই", "দিকে", "দিতে", "দিয়ে", "দিয়েছে", "দিয়েছেন", "দিলেন", "দু", "দুটি", "দুটো", "দেয়", "দেওয়া", "দেওয়ার", "দেখা", "দেখে", "দেখতে", "দ্বারা", "ধরে", "ধরা", "নয়", "নানা", "না", "নাকি", "নাগাদ", "নিতে", "নিজে", "নিজেই", "নিজের", "নিজেদের", "নিয়ে", "নেওয়া", "নেওয়ার", "নেই", "নাই", "পক্ষে", "পর্যন্ত", "পাওয়া", "পারেন", "পারি", "পারে", "পরে", "পরেই", "পরেও", "পর", "পেয়ে", "প্রতি", "প্রভৃতি", "প্রায়", "ফের", "ফলে", "ফিরে", "ব্যবহার", "বলতে", "বললেন", "বলেছেন", "বলল", "বলা", "বলেন", "বলে", "বহু", "বসে", "বার", "বা", "বিনা", "বরং", "বদলে", "বাদে", "বার", "বিশেষ", "বিভিন্ন", "বিষয়টি", "ব্যবহার", "ব্যাপারে", "ভাবে", "ভাবেই", "মধ্যে", "মধ্যেই", "মধ্যেও", "মধ্যভাগে", "মাধ্যমে", "মাত্র", "মতো", "মতোই", "মোটেই", "যখন", "যদি", "যদিও", "যাবে", "যায়", "যাকে", "যাওয়া", "যাওয়ার", "যত", "যতটা", "যা", "যার", "যারা", "যাঁর", "যাঁরা", "যাদের", "যান", "যাচ্ছে", "যেতে", "যাতে", "যেন", "যেমন", "যেখানে", "যিনি", "যে", "রেখে", "রাখা", "রয়েছে", "রকম", "শুধু", "সঙ্গে", "সঙ্গেও", "সমস্ত", "সব", "সবার", "সহ", "সুতরাং", "সহিত", "সেই", "সেটা", "সেটি", "সেটাই", "সেটাও", "সম্প্রতি", "সেখান", "সেখানে", "সে", "স্পষ্ট", "স্বয়ং", "হইতে", "হইবে", "হৈলে", "হইয়া", "হচ্ছে", "হত", "হতে", "হতেই", "হবে", "হবেন", "হয়েছিল", "হয়েছে", "হয়েছেন", "হয়ে", "হয়নি", "হয়", "হয়েই", "হয়তো", "হল", "হলে", "হলেই", "হলেও", "হলো", "হিসাবে", "হওয়া", "হওয়ার", "হওয়ায়", "হন", "হোক", "জন", "জনকে", "জনের", "জানতে", "জানায়", "জানিয়ে", "জানানো", "জানিয়েছে", "জন্য", "জন্যওজে", "জে", "বেশ", "দেন", "তুলে", "ছিলেন", "চান", "চায়", "চেয়ে", "মোট", "যথেষ্ট", "টি"]

def __word_divider(sentence):
    word_dividers = [' ', ',', '.', ';', '"', "'", '`', '(', ')', '[', ']', '-', '‘', '’‌', '%', '/', '\\']
    # idk what this regex part is doing. got it from stack overflow.
    divider_pattern = '|'.join(map(re.escape, word_dividers))
    tokenizer = RegexpTokenizer(f'[^{divider_pattern}]+|[{divider_pattern}]')
    tokens = tokenizer.tokenize(sentence)

    # removes the divider tokens and also strips them
    processed_tokens = []
    for token in tokens:
        if token.strip() and token not in word_dividers:
            processed_tokens.append(token.strip())

    refined_wordlist = [word for word in processed_tokens if word not in stopwords]

    return refined_wordlist


@lru_cache(maxsize=1)
def __model_loader():
    fasttext.util.download_model('bn', if_exists='ignore')
    ft = fasttext.load_model('cc.bn.300.bin')
    return ft


def __vectorizer(sentence):
    model = __model_loader()

    word_list = __word_divider(sentence)
    vectors_of_words_in_sentence = []
    for word in word_list:
        vector = model.get_word_vector(word)
        vectors_of_words_in_sentence.append(vector)

    return vectors_of_words_in_sentence


def __euclidean_distance(vect1, vect2):
    return numpy.linalg.norm(vect1 - vect2)


def wgss(sentence1, sentence2, sigma=0.70710678):
    word_set_1 = __vectorizer(sentence1)
    word_set_2 = __vectorizer(sentence2)

    D = []

    # finding the distance of the closest word from sentence 1 for every word in sentence 2
    for word_i in word_set_1:
        D_msw = float('inf')
        for word_j in word_set_2:
            distance = __euclidean_distance(word_i, word_j)
            D_msw = min(D_msw, distance)
        D.append(D_msw)

    # finding the distance of the closest word from sentence 2 for every word in sentence 1
    for word_j in word_set_2:
        D_msw = float('inf')
        for word_i in word_set_1:
            distance = __euclidean_distance(word_j, word_i)
            D_msw = min(D_msw, distance)
        D.append(D_msw)

    # to avoid zero divides by chance
    if len(D) == 0:
        return 0

    n = len(D)

    sum_D_square = 0
    for i in range(n):
        sum_D_square += D[i] ** 2

    similarity = numpy.exp(- sum_D_square / (2 * n * sigma ** 2))
    return similarity

    
