
# //////////////////////////////////////////////////////////////////////
# ///////////// CHECK THE TWEETGRABBER REPO FOR DATASETS //////////////
# ////////////////////////////////////////////////////////////////////

import re
import numpy as np
import pandas as pd
import os
import glob
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim_models  # don't skip this
# import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# Read in data

globStr = "*"
dirName = "/home/tommy/PycharmProjects/nlpTwitter/apr_8_hash/"
data_files = [(x[0], x[2]) for x in os.walk(dirName)]
# print(data_files[0][1])
tweetList = []
for file in data_files[0][1]:
    docFile = open(dirName + file, "r")
    doc = docFile.read()
    entry = [file, doc]
    tweetList.append(entry)

tweetList = np.array(tweetList)
# for theFile in glob.glob(os.path.join(self.dirName, globStr)):
#     with open(theFile, mode='r', encoding='cp1252') as f:  # open in readonly mode
#         docText = theFile.read()


# filePath = "apr_8_hash/" + "#StrangerThings4.txt"
# df = pd.DataFrame(tweetList)
# print(df.target_names.unique())

# Remove new line characters
for tweetIdx, tweet in enumerate(tweetList):
    test = tweet
    tweetList[tweetIdx][1] = re.sub('\s+', ' ', tweet[1])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


# tst2 = [row[1] for row in tweetList]

tst2 = tweetList[:, 1]

data_words = list(sent_to_words(tweetList[:, 1]))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1])

# Human readable format of corpus (term-frequency)
id2word[0]
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

print("Done!")