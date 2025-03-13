
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet

s_words = stopwords.words('english')

t1= pd.read_csv('rem_emoji.csv',index_col=False)
t1=t1.drop(t1.columns[[0]],axis=1)
t1['rem_emojis']=t1['rem_emojis'].astype(str)

#url=re.compile("([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
#t1["rem_url"]=t1["rem_emojis"].str.replace(url,"")

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
t1['rem_url'] = [remove_url(tweet) for tweet in t1['rem_emojis']]

#removing numbers
t1['rem_url1']= t1['rem_url'].str.replace('\d+', '')

#coverting to lower case

t1["lower"]= t1["rem_url1"].str.lower()


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
t1.to_csv('rem_url.csv')