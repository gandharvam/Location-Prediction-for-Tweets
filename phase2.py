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
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Conv1D,MaxPooling1D,Activation,Concatenate,Reshape


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


print("No of tweets in dataset : ",len(X))

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


# transformer encoder implemented as layer
# sandwhich transformer 

class TransformerBlock(layers.Layer):
  def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
    super().__init__() # for stop referring the base class 
    self.att = layers.MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim)
    self.ffn = keras.Sequential([layers.Dense(ff_dim,activation='relu'),layers.Dense(embed_dim,)])
    #print("hello ")
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)

  def call(self,inputs):
    # sandwhich transformer ( sssfsfff )
    attn_output = self.att(inputs, inputs)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(inputs + attn_output)
    attn_output = self.att(out1, out1)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(inputs + attn_output)
    attn_output = self.att(out1, out1)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out1 = self.layernorm2(out1 + ffn_output)
    attn_output = self.att(out1, out1)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out1 = self.layernorm2(out1 + ffn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out1 = self.layernorm2(out1 + ffn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    return self.layernorm2(out1 + ffn_output)
    
    




# Token and Position Embedding 

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



 # character level model

# creating the character vocublary

num = 1

char_vocab = {}

for line in X:
  for j in range(0,len(line)):
    if line[j] not in char_vocab:
      char_vocab[line[j]] = num
      num = num + 1



char_vocab_size = len(char_vocab) + 1


# changing each char to its corresponding index in vocublary and then apply padding to get max length = 140

x_char = []
for i in range(0,len(X)):
  text = X[i]
  seq = [char_vocab[char] for char in text]
  
  seq = pad_sequences([seq], maxlen=140)[0]
  #print(len(seq))
  x_char.append(seq)


x_char = np.array(x_char)
x_char_train,x_char_test = x_char[:5000,:],x_char[5000:,:]


''' hyperparameters '''
maxlen1 = 33
embed_dim1 = 200
embed_dim2 = 100
num_heads1 = 10
num_heads2 = 8
ff_dim1 = 50
ff_dim2 = 50
maxlen2 = 140
filter_num = 32
filter_size = 3
pooling_size = 6
stack = 2
filter_sizes = [3,4,5]


# left side model
inputs1 = layers.Input(shape=(maxlen1,))
embedding_layer1 = TokenAndPositionEmbedding(maxlen1,word_vocab_size,embed_dim1)
x1 = embedding_layer1(inputs1)


transformer_block1 = TransformerBlock(embed_dim1,num_heads1,ff_dim1)
x1 = transformer_block1(x1)
x1 = layers.GlobalAveragePooling1D()(x1)
x1 = layers.Dropout(0.1)(x1)

# right side model

embedding_layer2 = Embedding(char_vocab_size,embed_dim2,input_length=maxlen2)
inputs2 = layers.Input(shape=(maxlen2,))
x2 = embedding_layer2(inputs2)
'''x2 = Conv1D(64,3,padding='SAME',
                        activation='relu',
                         use_bias=True)(x2)
x2 = MaxPooling1D(pool_size=6)(x2)'''

# parallel convolution layer
conv_0 = Conv1D(64,filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(x2)
conv_1 = Conv1D(64,filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(x2)
conv_2 = Conv1D(64,filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(x2)

maxpool_0 = MaxPooling1D(pool_size=2*filter_sizes[0], padding='valid')(conv_0)
maxpool_1 =  MaxPooling1D(pool_size=2*filter_sizes[1], padding='valid')(conv_1)
maxpool_2 =  MaxPooling1D(pool_size=2*filter_sizes[2], padding='valid')(conv_2)


x2 = Concatenate(axis=1)([maxpool_0,maxpool_1,maxpool_2])
transformer_block2 = TransformerBlock(64,num_heads2,ff_dim2)
  
x2 = transformer_block2(x2)
  #print(x2.shape)
x2 = layers.GlobalAveragePooling1D()(x2)
x2 = layers.Dropout(0.1)(x2)


# final model

x3 = Concatenate()([x1, x2])


# adding extra layer 
covar2 = Reshape((1,264))(x3)

transformer_block3 = TransformerBlock(264,10,50);
x3 = transformer_block3(covar2)
x3 = Reshape((264,))(x3)

outputs1 = Dense(no_of_city, activation='softmax',name='city')(x3)
outputs2 = Dense(num_of_country, activation='softmax',name='country')(x3)
model = Model(inputs=[inputs1,inputs2], outputs=[outputs1,outputs2])



model.summary()

plot_model(model, to_file='phase2_model.png', show_shapes=True, show_layer_names=True)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print("Training .......................... ")

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience = 3) # early stopping   
model.fit([x_word_train,x_char_train],[y1_train,y2_train], batch_size=32, epochs=10,verbose = 1,callbacks = [callback])

print("Testing .......................... ")

arr = model.evaluate([x_word_test,x_char_test], [y1_test,y2_test], verbose = 1, batch_size = 32)


print("Accuracy for predicting country : ",arr[4])
print("Accuracy for predicting city : " , arr[3])         
          
