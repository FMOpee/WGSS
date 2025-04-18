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
1. Sentence similarity (returns the similarity between two sentences). ***This is the main contribution of the paper***
2. Bengali summarizer (returns an extractive summary of an input)

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
WGSS.wgss_sentence_similarity(sentence_1, sentence_2, sigma=0.70710678)
```
The input is two sentence and a control variable sigma, the output is a similarity score between the two sentences ranging from 0 to 1.
*Note: First run takes about 15 minute to download and load the vector model. It will be cached so later runs will be instantaneous*

```python
WGSS.wgss_bengali_summary(input_document, sigma=0.70710678, proportion=0.15)
```
The input takes a string to summarize. optional inputs include sigma, the control variable for sentence similarity calculation, and proportion, the size of the expected summary compared to the input document.

*Note: First run takes about 15 minute to download and load the vector model. It will be cached so later runs will be instantaneous*
