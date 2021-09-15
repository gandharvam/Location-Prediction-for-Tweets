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
import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

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

VALIDATION_SPLIT = 0.20
EMBEDDING_DIM = 100 
embedding_dim = 100 
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
batch_size = 30
epochs = 2
MAX_SEQUENCE_LENGTH = 33


embedding_layer = Embedding(word_vocab_size,
                            EMBEDDING_DIM,
                            input_length=33,
)

inputs = Input(shape=(33,), dtype='int32')
embedding = embedding_layer(inputs)

print(embedding.shape)
reshape = Reshape((33,EMBEDDING_DIM,1))(embedding)
print(reshape.shape)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=num_of_country, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_word_train, y2_train, batch_size=32, epochs=10, verbose=1)
score,acc = model.evaluate(x_word_test, y2_test, verbose = 1, batch_size = 32)

print("acc: %.2f" % (acc))
plot_model(model, to_file='model_plot.png')




