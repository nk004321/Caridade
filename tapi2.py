#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import re
from sqlalchemy.exc import ProgrammingError

import json
import pandas as pd
from datafreeze import freeze
import dataset
from dataset import *

import tkinter as tkr
from tkinter import filedialog

top = tkr.Tk()
top.geometry("600x600")




from tweepy.streaming import StreamListener
from tweepy import Stream
    # Enter Twitter API Keys
access_token = "926853440868728832-DaxDJdhK1izJnXQ6p5SMlkeYInIYeob"
access_token_secret = "jY7qgNhhy4OnbfPrK4VkOTRBrApy8MXov128kXJX96DC5"
consumer_key = "aUlQu5h5gA6X1MLiR505r4yDu"
consumer_secret = "qdQmbZVRbyLOt5Kw029OdgLfqoxJNvm1MtnDxc7rQxd2AhfW05"
    
    
    # Create tracklist with the words that will be searched for
tracklist = ['charity','hospital','urgent']
    # Initialize Global variable
tweet_count = 0
    # Input number of tweets to be downloaded
n_tweets = 200
a=""
     # Create the class that will handle the tweet stream
class StdOutListener(StreamListener):
        
        
         
            
              
            def on_data(self, data):
                
                global tweet_count
                global n_tweets
                global stream
                global a
                if tweet_count < n_tweets:
                
                    
                    a=a+data
                    tweet_count += 1
                    return True
                else:
                    stream.disconnect()
        
            def on_error(self, status):
                print(status)
    
        # Handles Twitter authetification and the connection to Twitter Streaming API
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)
stream.filter(track=tracklist,languages=["en"])
def twitter_scraping():
            
    
    
    #Converting to dataframe
    print(type(a))
    
    
    with open('tweet_data.txt', 'w') as f:
        f.write(a)  
    
    f.close()
    tweets_data_path = "tweet_data.txt" 
    
    tweets_data = []  
    tweets_file = open(tweets_data_path, "r")  
    for line in tweets_file:  
        try:  
            tweet = json.loads(line) 
            tweets_data.append(tweet)  
        except:  
            continue
    print(tweets_data)
    tweets = pd.DataFrame()
    tweets['text'] = list(map(lambda tweet: tweet['text'],tweets_data ))
    tweets['Username'] = list(map(lambda tweet: tweet['user']['screen_name'],tweets_data ))
    tweets['Timestamp'] = list(map(lambda tweet: tweet['created_at'], tweets_data))
    
    print(tweets)
    #Dropping duplicates
    tweets.drop_duplicates(subset="text",keep="first",inplace=True)
    #Storing as csv
    tweets.to_csv('new_tweets.csv')
    lab=tkr.Label(top,text='Twitter Data Successfully Scraped')
    lab.pack()
    


def data_cleaning():

    t1= pd.read_csv('new_tweets.csv',index_col=False)
    t1=t1.drop(t1.columns[[0]],axis=1)
    
    
    """def rem_username(txt):
     return  re.sub('@[^\s]+','',txt)"""
    #Removing emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]")
    
    t1['rem_emojis'] = t1['text'].str.replace(emoji_pattern, '')
    #Removing usernames from scraped tweets
    #t1['rem_emojis']= [rem_username(tweet) for tweet in t1['rem_emojis']]
    t1['rem_emojis']=t1['rem_emojis'].str.replace('@[^\s]+','')
    t1['rem_emojis']=t1['rem_emojis'].str.replace('RT','')
    print(t1)
    
    #Removing comtractions t1['rem_con']=[contractions.fix(word) for word in t1['rem_emoji']]
    
    
    t1.to_csv('new_rem_emoji.csv')



    import string
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords,wordnet
    
    s_words = stopwords.words('english')
    
    t1= pd.read_csv('new_rem_emoji.csv',index_col=False)
    t1=t1.drop(t1.columns[[0]],axis=1)
    t1['rem_emojis']=t1['rem_emojis'].astype(str)
    
    #url=re.compile("([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    #t1["rem_url"]=t1["rem_emojis"].str.replace(url,"")
    
    def remove_url(txt):
        return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
    t1['rem_url'] = [remove_url(tweet) for tweet in t1['rem_emojis']]
    
    #removing numbers
    t1['rem_url']= t1['rem_url'].str.replace('\d+', '')
    
    #coverting to lower case
    
    t1["lower"]= t1["rem_url"].str.lower()
    t1.dropna(subset = ["lower"], inplace=True)
    
    #removing stopwords
    t1['no_stopw'] = t1['lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (s_words)]))
    
    #tokenizing
    
    t1['tokenized'] = t1['no_stopw'].apply(word_tokenize)
    
    #removing punctuation
    
    punc = string.punctuation
    t1['no_punc'] = t1['tokenized'].apply(lambda x: [word for word in x if word not in punc])
    
    #POS tagger
    
    t1['pos_tags'] = t1['no_punc'].apply(nltk.tag.pos_tag)
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    t1['wordnet_pos'] = t1['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    
    #lemmatization
    
    lemmatizer = WordNetLemmatizer()
    t1['lemmatized'] = t1['wordnet_pos'].apply(lambda x: [lemmatizer.lemmatize(word, tag) for word, tag in x])
    
    
    print(t1)
    t1.to_csv('new_rem_url.csv')
    lab=tkr.Label(top,text='Dataset Cleaned Successfully')
    lab.pack()





    
def actual_clustering():
    
        #clustering model
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

    cls = MiniBatchKMeans(n_clusters=11, random_state=10)
    cls.fit(b)
        
        #predicting
    cls.predict(c)
        
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
    lab=tkr.Label(top,text='Clustering  Successfully Done')
    lab.pack()
   
def topic_modeling():

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
    a1=cls.transform(b).argsort(axis=1)[:,-1]  
    topics=pd.DataFrame({'Labels':a1})
    t4=pd.read_csv('rem_url.csv',index_col=False)
    topics['Username']=t4['Username']
    
    topics['Entire_text']=t4['rem_url']
    topics["Timestamp"]=t4['Timestamp']
    topics.to_csv('topics.csv')
    
    
    
    
    lab=tkr.Label(top,text='Topic Modeling Successfully Performed')
    lab.pack()
    
def joining_tables():
    #adding new tweets to existing dataframe
    t1= pd.read_csv('main_db.csv',index_col=False)
    t1=t1.drop(t1.columns[[0]],axis=1)
    print(t1)
    tweets=pd.read_csv('new_tweets.csv',index_col=False)
    tweets=tweets.drop(tweets.columns[[0]],axis=1)
    print(tweets)
    t1=t1.append(tweets,ignore_index=True)
    t1.drop_duplicates(subset="text",keep="first",inplace=True)
    print(t1)
    t1.to_csv('main_db.csv')
    
    #adding new_rem_emoji to rem_emojis
    t2= pd.read_csv('rem_emoji.csv',index_col=False)
    t2=t2.drop(t2.columns[[0]],axis=1)
    
    tweets2=pd.read_csv('new_rem_emoji.csv',index_col=False)
    #tweets2=tweets.drop(tweets2.columns[[0]],axis=1)
    
    t2=t2.append(tweets2,ignore_index=True)
    t2.to_csv('rem_emoji.csv')
    
    #adding new_rem_url to rem_url
    t3= pd.read_csv('rem_url.csv',index_col=False)
    t3=t3.drop(t3.columns[[0]],axis=1)
    
    tweets3=pd.read_csv('new_rem_url.csv',index_col=False)
    tweets3=tweets3.drop(tweets3.columns[[0]],axis=1)
    
    t3=t3.append(tweets3,ignore_index=True)
    t3.to_csv('rem_url.csv')
    lab=tkr.Label(top,text='Newly scraped data added to existing dataframe')
    lab.pack()
    
    
#Defining and packing the buttons
B1 = tkr.Button(top, text ="Scrap Data from Twitter", command = twitter_scraping)

B2 = tkr.Button(top,text="Perform Data Cleaning",command=data_cleaning)

B4 = tkr.Button(top,text='Perfom Clustering',command=actual_clustering)
B5 = tkr.Button(top,text="Perform Topic Modeling",command=topic_modeling)
B6 = tkr.Button(top,text="Add scraped data to main Dataframe",command=joining_tables)

B1.pack(pady=10)
B2.pack(pady=20)

B4.pack(pady=20)
B5.pack(pady=20)
B6.pack(pady=20)
top.mainloop()