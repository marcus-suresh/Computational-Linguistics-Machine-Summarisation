# Copyright (C) 2019 Suresh, Marcus <marcus.suresh@industry.gov.au>
# License: CC-BY, marcus.suresh@industry.gov.au

"""

Topic Modelling using spaCy

"""

from setuptools import setup

def readme():
	with open('README.md') as f:
		README = f.read()
		return README

setup(
	name="TM_spaCy"
	version="1.0.0"
	author="Marcus Suresh"
	author_email="marcus.suresh@industry.gov.au"
	license="CC-BY"
	classifiers=[
		"Programming Language :: Python ::3"
		"Programming Language :: Python :: 3.7"
	],
	description="An lightweight, Topic Modeller using spaCy"
	long_description=readme()	
)

###20180628 - Python Code for Text Cleaning###
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
	
###20180628 - Python Code for importing NLTKs Wordnet###
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
	
###Filtering out stopwords###
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

Define a function to prepare the text for topic modelling:
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

	import random
	
text_data = []
with open('dataset.csv') as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            print(tokens)
            text_data.append(tokens)
			
			
###LDA with Gensim creating a dictionary from the data then converting it to a bag-of-words###
from gensim import corpora
dictionary = corpora.Dictionary(text_data)corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

###pyLDAvis###
pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. 
The package extracts information from a fitted LDA topic model to inform an interactive web-based visualisation. 

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
