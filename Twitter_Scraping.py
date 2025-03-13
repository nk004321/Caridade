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

