# Word-pair-based Gaussian Sentence Similarity

This repository is the official implementation of the paper "[A Novel Word Pair-based Gaussian Sentence Similarity Algorithm For Bengali Extractive Text Summarization](https://doi.org/10.48550/arXiv.2411.17181)".

## citation

```
@article{morshed-2024-wgss,
  title={A Novel Word Pair-based Gaussian Sentence Similarity Algorithm For Bengali Extractive Text Summarization},
  author={Morshed, Fahim and Rahman, Md Abdur and Ahmed, Sumon},
  journal={arXiv preprint arXiv:2411.17181},
  year={2024}
}
```

## Functionality

This package includes the following functionality:
1. Bengali preprocessor (tokenizes a document and removes stopword)
2. Bengali stopwords (returns a list of bengali stopwords)
3. Curated Bengali Summarization Dataset (returns 250 articles each with 2 summaries curated by two different person)
4. Sentence vectorizer (replaces each word of a sentence with their word vector embedding using fasttext)
5. Sentence similarity (returns the similarity between two sentences). ***This is the main contribution of the paper***
6. Bengali summarizer (returns an extractive summary of an input)

## Installation

In colab/jupyter use this to install or import the package
```python
!git clone https://www.github.com/FMOpee/WGSS.git
!pip install WGSS/
import WGSS
```

Or simply use this in terminal
```sh
git clone https://www.github.com/FMOpee/WGSS.git
pip install WGSS/
```
and then use import
```python
import WGSS
```

## Usage

The package has the following functions that can be used:
```python
WGSS.preprocessing(input_document)
```
Here the input has to be a string and the output will be a 2d list `[ [w11, w12, w13..], [w21, w22, w23,...],...]`
```python
WGSS.get_bengali_stop_words()
```
The output is a list of strings: `[ sw1, sw2, sw3,... ]`
```python
WGSS.get_curated_bengali_summarization_dataset()
```
The output is a list of dictionary objects. `[ {"document":"....", "summary-1":"...", "summary-2":"..."}, {"document":"....", "summary-1":"...", "summary-2":"..."},... ]`
```python
WGSS.vectorize(list_of_words_in_a_sentence)
```
The input is a list of words in a sentence and the output is a list of word embedding vectors obtained from fasttext.

*Note: First run takes about 30 minute to download and load the vector model. It will be cached so later runs will be instantaneous* 
```python
WGSS.sentence_similarity(sentence_i, sentence_j, sigma=5e-11)
```
The input is two sets of vectors gotten from the vectorize section and a control variable sigma, the output is a similarity score between the two sentences ranging from 0 to 1.
```python
WGSS.get_bengali_summary(input_document, sigma=5e-11, proportion=0.2)
```
The input takes a string to summarize. optional inputs include sigma, the control variable for sentence similarity calculation, and proportion, the size of the expected summary compared to the input document.

*Note: First run takes about 30 minute to download and load the vector space. It will be cached so later runs will be instantaneous*
