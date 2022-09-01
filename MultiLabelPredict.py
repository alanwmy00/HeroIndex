#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

tf.random.set_seed(1)


import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
nltk.download('stopwords')
stops = set(stopwords.words('english'))
stops.add("'s")
stops.remove("not")

def clean(sentence):
    # delete stopwords
    temp = " ".join(filter(lambda x: x not in stops, sentence.split()))
    # Remove punctuation
    temp = temp.translate(str.maketrans('', '', punctuation))
    # remove non-english characters
    temp = temp.encode("ascii", "ignore").decode()
    # Change all to lower case
    temp = temp.lower()
    # Delete numbers
    temp = re.sub(r'[0-9]', "", temp)
    # Delete excessive spaces and return
    return re.sub("  ", " ", temp)


articles = pd.read_csv("data/Article-2022-05-08.csv")
articles_labeled = articles.copy()
articles_labeled['key_terms'] = articles['key_terms'].apply(lambda s: [l for l in str(s).split(',')])
mlb = MultiLabelBinarizer()
mlb.fit(articles_labeled["key_terms"])
y_bin = mlb.transform(articles_labeled["key_terms"])
articles_labeled['text'] = articles_labeled['text'].fillna("")
articles_labeled['text'] = articles_labeled['title'] + articles_labeled['text']
articles_labeled["text"] = articles_labeled["text"].apply(clean)


data = tf.data.Dataset.from_tensor_slices((articles_labeled["text"], y_bin))
data = data.shuffle(buffer_size = len(data), seed=1)

train_size = int(0.7*len(data))
val_size   = int(0.1*len(data))

train = data.take(train_size)
val   = data.skip(train_size).take(val_size)
test  = data.skip(train_size + val_size)


# only the top distinct words will be tracked
max_tokens = 2000

# each headline will be a vector of length 25
sequence_length = 25

vectorize_layer = TextVectorization(
    max_tokens=max_tokens, # only consider this many words
    output_mode='int',
    output_sequence_length=sequence_length) 

headlines = train.map(lambda x, y: x)
vectorize_layer.adapt(headlines)


new_model = tf.keras.models.load_model('saved_model/classfication_model_multi_label.h5')


keys = dict(zip(range(16), [int(i) for i in ['1', '10', '11', '12', '13', '14', '15', '16', '2', '3', '4', '5',
       '6', '7', '8', '9']]))

def idx_to_label(arr):
    return [[keys[i] for i in range(16) if j[i] == 1] for j in arr]

def predict(x):
    """
    predict labels
    @params x: 1d or 2d array or list
    @return labels in a 2d array
    """
    x = np.array(x).reshape(-1,)
    x = np.array(list(map(clean, x)))
    x = tf.expand_dims(x, -1)
    return idx_to_label(np.round(new_model.predict(vectorize_layer(x))))  



print(predict([
    """
What’s next for Facebook, now Meta, when Sheryl Sandberg exits? It has grown over the years from a start-up, college project under Mark Zuckerberg, to one of the world’s largest, most powerful, successful and influential communications platforms which took both leaders to achieve. Now that Sandberg is calling it quits, how will Zuckerberg fill her role and what can we expect from Facebook going forward.
Zuckerberg was the creative genius and Sandberg was the executive leader. So, removing Sandberg from the equation is important. One way or another, things will change.
Over time, the same thing has happened time and time again. Think of the retirement of Bernie Marcus and Arthur Blank of the Home Depot. Or the leaving of Bill Gates, Steve Ballmer and Paul Allen of Microsoft. Or Steve Jobs, Steve Wozniak and John Sculley of Apple.
Yes, things will change. Things always change. Whether they be for the better or worse is the only question.
What does the future of Facebook look like?
When a successful leader steps down and is replaced by a new one, many times companies become lost until they hopefully, ultimately find the right replacement leader. Over time, we have seen many companies succeed and many others continue to struggle. Think Apple, Google, Amazon, Twitter, Motorola, Blackberry, GE, IBM and so many others.
Success is a mixed bag at best. It will either turn the entire organization into an unsteady mess, or it will be a strong step in the next generation. We’ve seen both play out.
Will Facebook be stronger or weaker after Sandberg leaves?
Will Facebook be stronger or weaker one year from today? Five years from today. That is the real question.
It often takes more than one genius to make a company thrive. Especially companies on the cutting edge. Different executives have different strengths and weaknesses. Very few have it all.
Over time, Facebook has grown from a simple way to keep college students connected under Zuckerberg to one of the biggest success stories of all time in social media, which itself is a new sector, under Sandberg.
Facebook struggles with privacy, security, tracking, personal info
Of course, even with their success, Facebook is not without its critics. They, and in fact every other social media company continually wrestle with privacy, security, protecting personal information, tracking and more.
That struggle will continue and will not likely be resolved anytime soon.
Both Zuckerberg and Sandberg have been needed to achieve their extraordinary level of success Facebook has achieved.
Zuckerberg has always been a genius on the entrepreneurial level. Sandberg was the genius on building the company.
End of one chapter in Facebook story, beginning of next
Now the big question is can Zuckerberg fill Sandberg’s shoes or will she have to be replaced?
Either way, this is the end of one chapter of the Facebook story. It’s time to start the next chapter.
Investors, workers and users are wondering whether this next chapter will be successful and growing or the beginning of a period of disorganization and struggle for the company.
Only time will tell.
Facebook investors, workers and users wonder what’s coming next?
Facebook has not been static over the years. They have grown by adding many new ideas and companies to the basic model with Instagram, Whatsapp, Oculus and more.
The traditional Facebook model was a success with rapid growth for many years. Then two things started to happen. Competition increased and traditional growth moderated.
Growth would come from other social media companies for users looking for something new. Sometimes young users were looking for something their parents were not using.
To answer this ongoing and growing threat, Facebook acquired many new competitors. So, growth strategy moved from creating the next successful app, to acquiring the competition.
Facebook acquired new competitors for growth
That has worked for years. That’s why I think this will continue.
Short-term, I expect to see Facebook operating the same as it has been with Sheryl Sandberg, at least for a while.
However, long term is the question. Every company, especially Facebook needs fresh blood and new ideas with a youthful exuberance that has always defined the company.
We don’t yet know what to expect from Facebook regarding long-term performance and growth.
We don’t know what to expect from Facebook going forward
The next big acquisition will be interesting when Zuckerberg no longer has Sandberg to debate with and create the successful path and move-forward strategy.
Going forward, things will be very interesting to watch at Facebook. So, the real question is will Sandberg be replaced or will Zuckerberg take the entire company on himself? Only time will tell.
Either way, this is the end of one chapter in the Facebook story and it is time to start the next one! I wish Zuckerberg success going forward writing this next chapter of the Facebook story.
        """
]))



