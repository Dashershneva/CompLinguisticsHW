from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import gensim
from gensim import corpora
import numpy as np
import re
from pprint import pprint


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        # print(", ".join([feature_names[i]
        #                  for i in topic.argsort()[::-1][:no_top_words]]))


n_topics = 20
# load the data, dataset description: http://scikit-learn.org/stable/datasets/twenty_newsgroups.html

with open("4_en.txt", 'r', encoding='utf-8') as text:
    dataset = text.readlines()
"""
dataset = fetch_20newsgroups(shuffle=True,
                             random_state=123,
                             remove=('headers', 'footers', 'quotes'))"""
#documents = dataset.data
#print('Text collection size and median length in symbols:')
#print(len(documents), np.median([len(d) for d in documents]))


tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                   min_df=10,
                                   ngram_range=(2,2),
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(dataset)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print()

nmf = NMF(n_components=n_topics)
nmf_doc_topic = nmf.fit_transform(tfidf)
print('NMF doc-topic shape:', nmf_doc_topic.shape)


# LDA on raw words counts
tf_vectorizer = CountVectorizer(max_df=0.8,
                                min_df=10,
                                ngram_range=(2,2),
                                stop_words='english')
tf = tf_vectorizer.fit_transform(dataset)
tf_feature_names = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_components=n_topics)
lda_doc_topic = lda.fit_transform(tf)
print('LDA doc-topic shape:', lda_doc_topic.shape)

no_top_words = 10
print('\nNMF top terms:')
display_topics(nmf, tfidf_feature_names, no_top_words)
print('\nLDA top terms:')
display_topics(lda, tf_feature_names, no_top_words)


# gensim LDA - может занять время

tok_collection = []     # токенизируем и чистим (аккуратно, а не как я) все документы коллекции
for d in dataset:
    tok_collection.append([w for w in re.split('[\W]+', d) if len(w) > 3])


dictionary = corpora.Dictionary(tok_collection)

corpus = [dictionary.doc2bow(text) for text in tok_collection]

ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                           num_topics=n_topics,
                                           id2word=dictionary)

pprint(ldamodel.print_topics())