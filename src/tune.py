import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

"""
Preprocessing data 

Mostly folllowing linkedin learning tutorial

Process:
 - lower case
 - recode mentions (@ 's) as @mention
 - recode links (https:\ or http:\) as ?link
 - replace "\n" (indicates new line) with space  
"""

text = train["text"].str.lower() # Make lower case
# Replace mentions with "@mention"
text = text.str.replace(r"(@\S+)", " @mention", regex=True)
# Replace link with "?link"
text = text.str.replace(r"(http://\S+)|(https://\S+)", " ?link", regex=True)
text = text.str.replace("\n", " ", regex=True) # Replace \n with space

def check_vocab(text):
    st = ""
    for t in text:
        st += (". " + t)
        
    words = pd.Series(st.split())
    words_vc = words.value_counts()
    
    words_vc.hist(bins=100)
    plt.show()
    
    print("There are", len(words_vc), "unique words")
    print("Of those words", sum(words_vc==1), "are only used once")

check_vocab(text)

vocab_size = 10000

def model_builder(hp):
    hp_embedding_size = hp.Int("num_words", min_value=24, max_value=60, step=6)
    
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, hp_embedding_size, input_length=30),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(20),
      tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model


# Split val and train
X_train, X_val, y_train, y_val = train_test_split(text, train["target"], 
                                              test_size=0.2, random_state=1)

    
tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<oov>")
tokenizer.fit_on_texts(X_train)

# First, find range of lengths. Helps determine max_len parameter
sequences = tokenizer.texts_to_sequences(X_train)

max_len = 30

def do_padding(sentences, tokenizer, maxlen, padding, truncating):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    
    return padded, sequences

padded_train, _ = do_padding(X_train, tokenizer, max_len, "post", "post")
padded_val, _ = do_padding(X_val, tokenizer, max_len, "post", "post")

import keras_tuner as kt

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(padded_train, y_train, epochs=20, 
             validation_data=(padded_val, y_val), callbacks=[stop_early])

tuner.get_best_hyperparameters()[0].get("num_words")

