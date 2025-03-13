
import pandas as pd
import re


t1= pd.read_csv('main_db.csv',index_col=False)
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


t1.to_csv('rem_emoji.csv')

