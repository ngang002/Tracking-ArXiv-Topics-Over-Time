# Importing standard Python Packages here 
import sqlite3
from tqdm import tqdm
import time

import numpy as np
import scipy as sp 
import pandas as pd 

import spacy
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import os
import multiprocessing 
from multiprocessing import Pool
NUM_CPUS = os.cpu_count()

from data_preProcessing import * 
from get_SimilarWords import * 


ARXIV_DATABASE_FILE = "arxive_astro_3000.db"
if __name__ == "__main__":
	# Connect to an existing database or create a new one if it doesn't exist
	con = sqlite3.connect(ARXIV_DATABASE_FILE)

	# Create a cursor object
	cur = con.cursor()

	#Extract the list of databases available
	cur.execute("PRAGMA database_list;")

	# Query the sqlite_master table for table names
	cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
	tables = cur.fetchall()

	if tables:
		print(f"Tables in '{ARXIV_DATABASE_FILE}':")
		for table_name in tables:
			print(table_name[0])  # Access the table name from the tuple
	else:
		print(f"No tables found in '{ARXIV_DATABASE_FILE}'.")

	#Read in all the different databases 
	databases = cur.fetchall()
	for db in databases: print(f"Database Name: {db[1]}, Path: {db[2]}")

	# Reading from arxiv_papers
	query = "SELECT * FROM arxiv_papers" 

	query = "SELECT *, STRFTIME('%Y', published) AS Year, STRFTIME('%m', published) AS Month, "+\
	        "LENGTH(authors) - LENGTH(REPLACE(authors, ',', ''))+1 as NumAuthors FROM arxiv_papers"



	# Convert the SQL database into a pandas dataframe
	df = pd.read_sql_query(query, con)
	df.drop(columns=['published', 'updated', 'citations', 'keywords', 'external_json'], inplace=True)
	cur.close()

	# Extract the abstract list from the Pandas dataframe and apply the preprocessing methods
	start_time = time.time()
	abstractList = df['abstract'].values
	with Pool(processes = NUM_CPUS) as p:
		cleanedAbstractList = p.map(preprocessTextPass1, abstractList)
	print(f"Time to preprocess abstract text: {time.time()-start_time:1.4f}")

	# Extracting all the words into a single list to check number of occurrences 
	vocabulary = getVocabulary(cleanedAbstractList)

	# Get all the words that we want to group together
	vocabulary_limitCut, vocabulary_OccurrenceDict = getVocabulary_Unique(vocabulary)

	# Find all the pairs of words that are similar to each other
	word_pairs = group_SimilarWords(vocabulary_limitCut, _NGRAM=2)
	print(f"Numer of Similarity Pairs: {len(word_pairs)}")

	# Get the canonical mapping of words togheter
	canonical_map = groupWords(word_pairs, vocabulary_OccurrenceDict)

	# Results
	# print(f"Number of Pairs: {len(canonical_map)}")
	# for _ in canonical_map: print(_, canonical_map[_])

	# Extract the abstract list from the Pandas dataframe and apply the preprocessing methods
	start_time = time.time()
	abstractList = df['abstract'].values
	with Pool(processes = NUM_CPUS) as p:
		cleanedAbstractList = p.starmap(replaceWords, [(cleanedAbstract, canonical_map) for cleanedAbstract in cleanedAbstractList])
	print(f"Time to preprocess abstract text: {time.time()-start_time:1.4f}")
	

	df['Cleaned Abstracts'] = cleanedAbstractList
	df.to_csv('cleaned_arxiv_astro.csv')



