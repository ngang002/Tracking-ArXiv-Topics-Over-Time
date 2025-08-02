import numpy as np
import pandas as pd 

from itertools import combinations
import networkx as nx

from data_preProcessing import *
from data_preProcessing import _WORDS

def get_ngrams(word, _NGRAM=3):
    word = f"#{word}#"  # padding to preserve start/end
    return set([word[i:i+_NGRAM] for i in range(len(word)-_NGRAM+1)])

def calc_JaccardSimilarity(a, b):
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0


def group_SimilarWords(vocabulary_limitCut, _NGRAM=2, _jacSim_THRESHOLD=0.5, _cosSim_THRESHOLD = 0.90):

    # Get a dictionary for all the n-grams for a words 
    word_ngrams = {w: get_ngrams(w, _NGRAM) for w in vocabulary_limitCut}

    embeddings = model.encode(vocabulary_limitCut)
    similarity_matrix = cosine_similarity(embeddings)

    # Calculate the Jaccard similarity between all the words. 
    # The Jaccard similarity measures the distance between two 
    # words by counting the number of similar n-grams, and normalizing
    # it by the all the possible pairs of n-grams.  If the Jaccard distance
    # is greater than some threshold, then that means that the words are 
    # similar enough.  
    SIMILAR_PAIRS = []
    for w1, w2 in combinations(np.arange(len(vocabulary_limitCut)), 2):
        word1, word2 = vocabulary_limitCut[w1], vocabulary_limitCut[w2]
        if 'mission' in word1 or 'mission' in word2: 
            continue
        # if 'galax' in w1 and 'galax' in w2:
        #     SIMILAR_PAIRS.append(('galaxy', w1))
        #     SIMILAR_PAIRS.append(('galaxy', w2))
        #     print("BOTH", w1, w2, SIMILAR_PAIRS[-1:])
        #     continue
        # elif 'galax' in w1:
        #     SIMILAR_PAIRS.append(('galaxy', w1))
        #     print("W1", w1, w2, SIMILAR_PAIRS[-1:])
        #     continue
        # elif 'galax' in w2:
        #     SIMILAR_PAIRS.append(('galaxy', w2))
        #     print("W2", w1, w2, SIMILAR_PAIRS[-1:])
        #     continue
        
        # Optional: quick length filter
        if abs(len(word1) - len(word2)) > 4:
            continue

        jacSim = calc_JaccardSimilarity(word_ngrams[word1], word_ngrams[word2])
        if (jacSim >= _jacSim_THRESHOLD and similarity_matrix[w1][w2] >= _cosSim_THRESHOLD) or (_jacSim_THRESHOLD >= 0.9): SIMILAR_PAIRS.append((word1, word2))

    return SIMILAR_PAIRS

def choose_OriginalWord(cluster, vocabulary_OccurrenceDict):
    cluster = list(cluster)
    # print(f"Length of the Cluster: {len(cluster)}")
    print(cluster)
    # score = list(map(_vocabulary_OccurrenceDict.get, cluster))
    word = []
    score = [] 
    for w in cluster: 
        word.append(w), score.append(vocabulary_OccurrenceDict[w])
    # print(word, score, np.argmax(score), score[np.argmax(score)], word[np.argmax(score)]) 
    return word[np.argmax(score)] #

def groupWords(_SIMILAR_PAIRS, vocabulary_OccurrenceDict):

    # Step 1: Build an undirected graph
    G = nx.Graph()
    G.add_edges_from(_SIMILAR_PAIRS)

    # Step 2: Extract connected components
    components = list(nx.connected_components(G))
    

    # Step 4: Build the mapping dict
    CANONICAL_MAP = {}
    for component in components:
        canonical = choose_OriginalWord(component, vocabulary_OccurrenceDict)
        for word in component:
            CANONICAL_MAP[word] = canonical

    print(f"Number of Categories: {len(components)}")
    print(f"Number of Pairs: {len(CANONICAL_MAP)}")

    return CANONICAL_MAP