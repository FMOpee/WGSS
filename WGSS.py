import numpy


def __euclidean_distance(vect1, vect2):
    return numpy.linalg.norm(vect1 - vect2)


def sentence_similarity(sentence_1, sentence_2, sigma=5e-11):
    D = []

    # finding the distance of the closest word from sentence 1 for every word in sentence 2
    for word_i in sentence_1:
        D_msw = float('inf')
        for word_j in sentence_2:
            distance = __euclidean_distance(word_i, word_j)
            D_msw = min(D_msw, distance)
        D.append(D_msw)

    # finding the distance of the closest word from sentence 2 for every word in sentence 1
    for word_j in sentence_2:
        D_msw = float('inf')
        for word_i in sentence_1:
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
