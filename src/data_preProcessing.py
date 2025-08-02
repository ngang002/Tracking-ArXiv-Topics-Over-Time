import numpy as np

import spacy
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load scientific/biomedical model if needed
# nlp = spacy.load("en_core_sci_sm")  # if using scispaCy
nlp = spacy.load("en_core_web_sm") #, disable=["ner", "parser"])  # faster for our use
_WORDS = list(nlp.vocab.strings) 
model = SentenceTransformer('allenai/scibert_scivocab_uncased')

def preprocessTextPass1(text):
    
    # Convert the abstract to lowercase
    text = text.lower()

    # Remove URLs, LaTeX math, inline citations, and other nonp-ascii texts
    text= re.sub(r"\d+", "", text)
    text = re.sub(r"\$.*?\$", "", text)  # remove LaTeX math and numbers
    text = re.sub(r"\[.*?\]|\(.*?et al\.\)", "", text)  # remove inline citations

    # Define a pattern
    latex_pattern = re.compile(r'''
    (\\[a-zA-Z]+\{[^}]*\})            | # commands with arguments e.g. citep{...}
    (\\[a-zA-Z]+)                     | # lone LaTeX commands e.g. \msun, \deg
    (\^\{[^}]*\})                     | # superscript math fragments e.g. ^{x}
    (\^\S+)                           | # other caret superscripts
    (\,\\~?[a-zA-Z]+)                 | # unit-like sequences: \,km, \,erg
    (\~\\?[a-zA-Z]+)                  | # tilde-prefixed units like ~\deg
    (\/\\?[a-zA-Z]+)                  | # slash-prefixed like /\beta_i
    (\{\\?[a-zA-Z]+\})                | # like {\deg}
    (\.\{\\?[a-zA-Z]+\})              | # .{\deg}
    (a\([a-z]+\)=\S*)                 | # a(li)=...
    (\\cite\w*\{[^}]*\})             | # citations
    (\\emph\{[^}]*\})                | # \emph{text}
    (\^\S*)                          | # leftover superscripts
    (\\[a-zA-Z]+\\?)                 | # generic backslash sequences like \kms\
    (\^\\?[a-zA-Z_]+)                | # ^alpha, ^m_sun
    (\{\\?[a-zA-Z]+\})               | # more embedded latex
    ''', re.VERBOSE)
    text = re.sub(latex_pattern, "", text)
    
    # Replace hyphens with spaces
    text = text.replace('-', '')

    # Combine NLTK and sklearn stopwords
    STOPWORDS = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)
    
    # Custom stopwords for astrophysics
    CUSTOM_STOPWORDS = set([
        "et", "al", "figure", "using", "based", "data", "datum", "analysis",
        "result", "results", "show", "use", "used", "paper", "new", "present", "study", "scientific",
        "tool", "dataset", "mass", "alpha", "beta", "article", "start", "stark", "end", "like"
    ])
    STOPWORDS.update(CUSTOM_STOPWORDS)
    
    # Tokenize + lemmatize with SpaCy
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        if token.lemma_ in STOPWORDS or token.lemma_ in string.punctuation:
            continue
        if len(token.lemma_) < 4:
            continue
        tokens.append(token.lemma_)

    return " ".join(tokens)

def preprocessTextPass2(text, CANONICAL_MAP):
    
    # Convert the abstract to lowercase
    text = text.lower()

    # Remove URLs, LaTeX math, inline citations, and other nonp-ascii texts
    text= re.sub(r"\d+", "", text)
    text = re.sub(r"\$.*?\$", "", text)  # remove LaTeX math and numbers
    text = re.sub(r"\[.*?\]|\(.*?et al\.\)", "", text)  # remove inline citations

    # Define a pattern
    latex_pattern = re.compile(r'''
    (\\[a-zA-Z]+\{[^}]*\})            | # commands with arguments e.g. \citep{...}
    (\\[a-zA-Z]+)                     | # lone LaTeX commands e.g. \msun, \deg
    (\^\{[^}]*\})                     | # superscript math fragments e.g. ^{x}
    (\^\S+)                           | # other caret superscripts
    (\,\\~?[a-zA-Z]+)                 | # unit-like sequences: \,km, \,erg
    (\~\\?[a-zA-Z]+)                  | # tilde-prefixed units like ~\deg
    (\/\\?[a-zA-Z]+)                  | # slash-prefixed like /\beta_i
    (\{\\?[a-zA-Z]+\})                | # like {\deg}
    (\.\{\\?[a-zA-Z]+\})              | # .{\deg}
    (a\([a-z]+\)=\S*)                 | # a(li)=...
    (\\cite\w*\{[^}]*\})             | # citations
    (\\emph\{[^}]*\})                | # \emph{text}
    (\^\S*)                          | # leftover superscripts
    (\\[a-zA-Z]+\\?)                 | # generic backslash sequences like \kms\
    (\^\\?[a-zA-Z_]+)                | # ^alpha, ^m_sun
    (\{\\?[a-zA-Z]+\})               | # more embedded latex
    ''', re.VERBOSE)
    text = re.sub(latex_pattern, " ", text)
    
    # Replace hyphens with spaces
    text = text.replace('-', ' ')

    # Combine NLTK and sklearn stopwords
    STOPWORDS = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)
    
    # Custom stopwords for astrophysics
    CUSTOM_STOPWORDS = set([
        "et", "al", "figure", "using", "based", "data", "datum", "analysis",
        "result", "results", "show", "use", "used", "paper", "new", "present", "study", "scientific",
        "tool", "dataset", "mass", "alpha", "beta", "article",
    ])
    STOPWORDS.update(CUSTOM_STOPWORDS)
    
    # Tokenize + lemmatize with SpaCy
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token in _CANONICAL_MAP.keys():
            print(f"We found a word in keys: {_}, replacing with {_CANONICAL_MAP[_]}")
            token = _CANONICAL_MAP[_]
            continue
        if token.is_stop or token.is_punct or token.like_num:
            continue
        if token.lemma_ in STOPWORDS or token.lemma_ in string.punctuation:
            continue
        if len(token.lemma_) < 4:
            continue
        tokens.append(token.lemma_)
    return " ".join(tokens)


def getVocabulary(abstractList):
    vocabulary = {}
    
    for ABSTRACT in tqdm(abstractList):
        tokenized_abstract = ABSTRACT.split(' ')
        for _ in tokenized_abstract:
            if _ not in vocabulary.keys():
                vocabulary[_] = 1 
            else: 
                vocabulary[_] += 1 
    
    return vocabulary

def getVocabulary_Unique(vocabulary):

    # Get all the unique words and count their occurrences in the text
    vocabulary_unique, occurrences = np.array(list(vocabulary.keys())), np.array(list(vocabulary.values())) # np.unique(vocabulary, return_counts=True)

    # Sorting the lists by the number of occurrences of each word, and selecting 
    # only those words which  occur more than N times.  However one could vary NCUT 
    # which dictates the number of times a word must occur or it will be cut. 
    NCUT = 5
    SORTED = np.argsort(occurrences)[::-1]
    vocabulary_unique, occurrences = vocabulary_unique[SORTED], occurrences[SORTED]
    SELECT = np.where(occurrences > NCUT)[0]
    vocabulary_limitCut = vocabulary_unique[SELECT]
    occurrences_limitCut = occurrences[SELECT]
    vocabulary_OccurrenceDict = {word: num for word, num in zip(vocabulary_limitCut, occurrences_limitCut)}
    print(f"Number of Unique Words: {len(vocabulary_unique)} \t Number of Words that occur >{NCUT} times: {len(vocabulary_limitCut)}")

    return vocabulary_limitCut, vocabulary_OccurrenceDict


def replaceWords(abstract, _CANONICAL_MAP):
    _TOKENIZED_ABSTRACT = abstract.split(' ')
    for _ in _TOKENIZED_ABSTRACT:
        if _ in _CANONICAL_MAP.keys():
            # print(f"We found a word in keys: {_}, replacing with {_CANONICAL_MAP[_]}")
            _ = _CANONICAL_MAP[_]
    
    return " ".join(_TOKENIZED_ABSTRACT)


def replaceWordsInAbstract(abstractList, CANONICAL_MAP): 

    print(abstract)
    for i, abstract in enumerate(abstractList):
        # print(f"Before: {ABSTRACT}")
        ABSTRACT = replaceWords(abstract, CANONICAL_MAP)
        # print(f"After: {ABSTRACT}")

    return abstractList