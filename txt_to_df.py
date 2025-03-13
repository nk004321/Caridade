import json
import pandas as pd
import re

t1= pd.read_csv('main_db.csv',index_col=False)
t1=t1.drop(t1.columns[[0]],axis=1)
print(t1)
tweets=pd.read_csv('new_tweets.csv',index_col=False)
tweets=tweets.drop(tweets.columns[[0]],axis=1)
print(tweets)

t1=t1.append(tweets,ignore_index=True)
print(type(t1))
#Removing Duplicates
t1.drop_duplicates(subset="text",keep="first",inplace=True)
print(t1)




#Converting to csv
t1.to_csv('main_db.csv')





