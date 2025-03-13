
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

t1= pd.read_csv('rem_url.csv',index_col=False)
t1=t1.drop(t1.columns[[0]],axis=1)

"""
output = []

for doc in t1['wordnet_pos']:
   for (word, pos_tag) in doc:
     if pos_tag=="n":
         output.append(word)
        
t1['nouns'] = output
print(t1.head())"""
#tfidf

vec = TfidfVectorizer()
vec.fit(t1['lemmatized'])

import pandas as pd
#a=pd.DataFrame(vec.transform(t1['lemmatized']).toarray(), columns=sorted(vec.vocabulary_.keys()))  print(a.head)
b=vec.transform(t1['lemmatized'])

#Topic modeling
n_topics = 5

from sklearn.decomposition import NMF
cls = NMF(n_components=n_topics, random_state=0)
cls.fit(b)

#Viewing the topics

# list of unique words found by the vectorizer
feature_names = vec.get_feature_names()

# number of most influencing words to display per topic
n_top_words = 15

for i, topic_vec in enumerate(cls.components_):
    print(i, end=' ')
    # topic_vec.argsort() produces a new array
    # in which word_index with the least score is the
    # first array element and word_index with highest
    # score is the last array element. Then using a
    # fancy indexing [-1: -n_top_words-1:-1], we are
    # slicing the array from its end in such a way that
    # top `n_top_words` word_index with highest scores
    # are returned in desceding order
    for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
        print(feature_names[fid], end=' ')
    print()
    
topics=pd.DataFrame(cls.transform(b).argsort(axis=1)[:,-1])
topics.to_csv('topics.csv')

