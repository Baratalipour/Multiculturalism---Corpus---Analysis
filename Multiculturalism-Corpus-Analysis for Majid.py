#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk

nltk.download('wordnet')

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Ensure nltk tokenizer works correctly with UTF-8 encoding
nltk.download('punkt')

# Set file path
file_path = '/Users/nasrin/Desktop/Multiculturalism_sylabes&research/Resourses/1/0.txt'

# Check if the file exists
if not os.path.exists(file_path):
    raise Exception(f"The file at the specified path does not exist: {file_path}")

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

# Preprocess the text
processed_docs = preprocess(text_data)

# Check if processed_docs is empty
if not processed_docs:
    raise ValueError("Preprocessed document is empty; check preprocessing steps.")

# Create a dictionary representation of the documents.
# If there is very little text, no_below should be 1 to ensure we have enough words.
dictionary = corpora.Dictionary([processed_docs])

# Create a bag of words
bow_corpus = [dictionary.doc2bow(processed_docs)]

# Check if the BoW corpus is empty
if not bow_corpus[0]:
    raise ValueError("Bag of words corpus is empty; the document may be too short for LDA modeling.")

# Setting up LDA
# With small text, we may want to reduce the number of topics.
num_topics = min(10, len(dictionary))  # Number of topics should not exceed number of unique tokens
if num_topics == 0:
    raise ValueError("The number of unique tokens is zero; cannot model topics.")

lda_model = models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)

# Get the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")


# In[ ]:




