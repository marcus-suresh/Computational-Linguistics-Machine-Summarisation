# Copyright (C) 2018-19 Suresh, Marcus <marcus.suresh@industry.gov.au>
# License: CC-BY, marcus.suresh@industry.gov.au

"""

Thompson-Reuters Sentiment Extractor

"""

from setuptools import setup

def readme():
	with open('README.md') as f:
		README = f.read()
		return README

setup(
	name="TRS_extractor"
	version="1.0.0"
	author="Marcus Suresh"
	author_email="marcus.suresh@industry.gov.au"
	license="CC-BY"
	classifiers=[
		"Programming Language :: Python ::3"
		"Programming Language :: Python :: 3.7"
	],
	description="An lightweight, low-code NLP sentiment extractor built for the Thompson Reuters Newswires platform"
	long_description=readme()	
)


###(1) Load NLTK Library and import corpus

from nltk.corpus import reuters
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
from nltk import word_tokenize  
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from operator import itemgetter
from pprint import pprint

### (2) Exploration and mining of Reuters Corpus

###(2.1)###Return the count of fileids###
documents = reuters.fileids()
print("Documents: {}".format(len(documents)))

###(2.2)###View training documents###
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
print("Total train documents: {}".format(len(train_docs_id)))
print(train_docs_id) ###Give list a name###

###(1.2)###View test documents###
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
print("Total test documents: {}".format(len(test_docs_id)))
print(test_docs_id) ###Give list a name###

###(1.3)###List of categories in train and test
categories = reuters.categories();
print(str(len(categories)) + " Categories within corpus")
reuters.categories(train_docs_id)
reuters.categories(test_docs_id)

###(1.4)###Query documents in a category
category_docs = reuters.fileids("jobs");
Queried_document_id = category_docs[15:20];
document_words = reuters.words(category_docs[10]);
print(document_words);  
print(reuters.raw(Queried_document_id));

###(1.5)Text cleanser###

###Tokenisation of text###
def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text));
	words = [word for word in words if word not in cachedStopWords]
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
	return filtered_tokens

###Return the representer, without transforming
def tf_idf(docs):	
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True, sublinear_tf=True);
	tfidf.fit(docs);
	return tfidf;

def feature_values(doc, representer):
	doc_representation = representer.transform([doc])
	features = representer.get_feature_names()
	return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

###(1.6)###Example of a document (with multiple labels)
doc_multilable = 'training/9865'
print(reuters.raw(doc_multilable))
print()
print(reuters.categories(doc_multilable))

# Documents per category.
category_distribution = [(category, len(reuters.fileids(category))) 
                         for category in categories]

category_distribution = sorted(category_distribution, 
                               key=itemgetter(0), 
                               reverse=True)

print("Most common categories")
pprint(category_distribution[:15])
print()

print("Least common categories")
pprint(category_distribution[-15:])
print()


train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

train_docs_id = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs_id = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Tokenisation 
vectorizer = TfidfVectorizer(cachedStopWords = cachedStopWords)

# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs_id)
vectorised_test_documents = vectorizer.transform(test_docs_id)

# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) 
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) 
                             for doc_id in test_docs_id])

# Classifier 
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
classifier.fit(vectorised_train_documents, train_labels)
predictions = classifier.predict(vectorised_test_documents)

print("Number of labels assigned: {}".format(sum([sum(prediction) 
                                                  for prediction in predictions])))

from sklearn.metrics import f1_score, precision_score, recall_score

# Show our quality
precision = precision_score(test_labels, predictions, average='micro')
recall = recall_score(test_labels, predictions, average='micro')
f1 = f1_score(test_labels, predictions, average='micro')
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, 
                                                                     recall, 
                                                                     f1))

precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, 
                                                                     recall, 
                                                                     f1))
