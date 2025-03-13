
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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


#clustering model

cls = MiniBatchKMeans(n_clusters=5, random_state=0)
cls.fit(b)

#predicting
cls.predict(b)

labels1=pd.DataFrame(cls.labels_)
labels1.to_csv('labels.csv')

# reduce the features to 2D
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(b.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)
##plotting
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(b))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


#evaluation using silhoutte score


print(silhouette_score(b, labels=cls.predict(b)))
#Evaluating
ks = range(2, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = MiniBatchKMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(b)
    
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
    
    print(silhouette_score(b, labels=model.predict(b)))
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()