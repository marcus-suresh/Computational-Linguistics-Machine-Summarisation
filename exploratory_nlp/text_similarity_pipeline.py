# Copyright (C) 2019 Suresh, Marcus <marcus.suresh@industry.gov.au>
# License: CC-BY, marcus.suresh@industry.gov.au

"""

Text Similarity Pipeline

"""

from setuptools import setup

def readme():
	with open('README.md') as f:
		README = f.read()
		return README

setup(
	name="semantic_similarities"
	version="1.0.0"
	author="Marcus Suresh"
	author_email="marcus.suresh@industry.gov.au"
	license="CC-BY"
	classifiers=[
		"Programming Language :: Python ::3"
		"Programming Language :: Python :: 3.7"
	],
	description="Text Similarity Pipeline"
	long_description=readme()	
)

# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

# Process whole documents
text = (u"CORPUS")
doc = nlp(text)

###Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

###Determine semantic similarities
doc1 = nlp(u"sentance 1")
doc2 = nlp(u"sentance 2")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)