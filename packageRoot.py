from .code.WgssSimilarity import wgss
from .code.Summarizer import get_summary

def wgss_sentence_similarity(sentence1, sentence2, sigma = 0.70710678):
    return wgss (sentence1,sentence2,sigma=sigma)

def wgss_bengali_summary (article, sigma = 0.70710678, proportion=0.15):
    return get_summary(article, sigma=sigma, proportion=proportion)
