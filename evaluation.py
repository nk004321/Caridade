import pandas as pd
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
    
t1= pd.read_csv('rem_url.csv',index_col=False)
t1=t1.drop(t1.columns[[0]],axis=1)
    
t2= pd.read_csv('new_rem_url.csv',index_col=False)
t2=t2.drop(t1.columns[[0]],axis=1)
    #tfidf

vec = TfidfVectorizer()
vec.fit(t1['lemmatized'])
    
vec2 = TfidfVectorizer()
vec2.fit(t1['lemmatized'])
    
    
    #a=pd.DataFrame(vec.transform(t1['lemmatized']).toarray(), columns=sorted(vec.vocabulary_.keys()))  print(a.head)
b=vec.transform(t1['lemmatized'])
c=vec2.transform(t2['lemmatized'])
ks = range(2, 15)
inertias = []
    
for k in ks:
            # Create a KMeans instance with k clusters: model
            model = MiniBatchKMeans(n_clusters=k,random_state=10)
            
            # Fit model to samples
            model.fit(b)
            
            
            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)
            
            print(k)
            print(silhouette_score(b, labels=model.predict(b)))
        
        # Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()