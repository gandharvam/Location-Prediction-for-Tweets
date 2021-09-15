import json
import preprocessor as p
import tensorflow as tf
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from keras.models import Sequential,Model
from keras.utils import to_categorical
from numpy import array
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Conv1D,MaxPooling1D,Activation,Concatenate
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention

data = []
with open('validation.tweet.json') as f:
    for line in f:
        data.append(json.loads(line))


city = dict()  # dictinary containing ( key : tweet_id , value : city name )
country = dict()  # dictinary containing ( key : tweet_id , value : country name )

for i in range(0,len(data)):
  tweet_id = data[i]['tweet_id']
  token = data[i]['tweet_city'].split('-')
  city[tweet_id] = token[0]
  country[tweet_id] = token[2]


data = []
with open('validation.tweet.json.IdTweet') as f:
    for line in f:
        data.append(json.loads(line))


tweet_text = dict()     # dictinary containing ( key : tweet_id , value : tweet_text )

''' applying preprocessing on tweet text such as -
1.removing emoji's
2.special symbol removing
3.converting to lowercase
4.removing urls
'''

table = str.maketrans('', '', string.punctuation)

p.set_options(p.OPT.URL, p.OPT.EMOJI,	p.OPT.NUMBER,	p.OPT.SMILEY)

for i in range(0,len(data)):
  tweet_id = data[i]['tweet_id']
  text = data[i]['text']
  text = p.clean(text)
  text = text.split()
  text = [word.lower() for word in text]
  # remove punctuation from each token
  text = [w.translate(table) for w in text]
  text =  ' '.join(text)
  if(len(text) > 0):
    tweet_text[tweet_id] = text


X = []  # tweet text
Y1 = [] #city
Y2 = [] #country


# X[i] , Y1[i] , Y2[i]  ------------> tweet_text,tweet_city,tweet_country 

for tweet_id in tweet_text.keys():
  X.append(tweet_text[tweet_id])
  Y1.append(city[tweet_id])
  Y2.append(country[tweet_id])


# max_length of the tweets

def max_length(X):
  return max(len(d.split()) for d in X)

max_length = max_length(X)
print('Max Description Length: %d' % max_length)



# we have kept only those words whose frequency is more than equals to 5

word_count_threshold = 5
word_counts = {}
nsents = 0
for sent in X:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

print('Vocublary size :  %d ' % len(vocab))


# to find the corresponding index for every wording in the vocublary
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ix += 1

#converting each tweet text word to its index and pading if required
x_word = []
for i in range(0,len(X)):
  text = X[i]
  seq = [wordtoix[word] for word in text.split(' ') if word in wordtoix]
  seq = pad_sequences([seq], maxlen=max_length)[0]
  x_word.append(seq)


# to convert city and country to one hot encoding
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder1.fit(Y2)
encoder2.fit(Y1)



y2 = encoder1.transform(Y2)
y1 = encoder2.transform(Y1)



num_of_country = np.max(y2) + 1
no_of_city = np.max(y1) + 1
y2 = keras.utils.to_categorical(y2, num_of_country)
y1 = keras.utils.to_categorical(y1, no_of_city)


# changing to numpy array
x_word = np.array(x_word)
y2 = np.array(y2)
y1 = np.array(y1)


# dividing data into training and test data
x_word_train,x_word_test = x_word[:5000,:],x_word[5000:,:]


y2_train,y2_test = y2[:5000,:],y2[5000:,:]
y1_train,y1_test = y1[:5000,:],y1[5000:,:]


word_vocab_size = ix


model = Sequential()
model.add(Embedding(word_vocab_size, 200, input_length=33, mask_zero=True))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(units=num_of_country, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_word_train, y2_train, batch_size=32, epochs=10, verbose=1)
score,acc = model.evaluate(x_word_test, y2_test, verbose = 1, batch_size = 32)
print("acc: %.2f" % (acc))
plot_model(model, to_file='model_plot.png')




