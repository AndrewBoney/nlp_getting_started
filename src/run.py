import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Input
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from src.utils import *
from model import (do_padding,process_keyword,get_location_cols, get_extra,
   location_dummies,preprocess_text,convert_cities,convert_countries)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Processing extra vars
keyword_bins = pd.read_csv("data/keyword_bins.csv", dtype={"keyword":str, 
                                                           "keyword_bin":str})
locations = pd.read_csv("data/locations.csv")

train_plus = train.merge(keyword_bins, how="left", on = "keyword"). \
    merge(locations, how="left", on="location")

train_plus.loc[train_plus["keyword_bin"].isna(), "keyword_bin"] = "missing"

train_plus = convert_cities(train_plus)
train_plus = convert_countries(train_plus)
train_plus = get_extra(train_plus)

dummies = pd.get_dummies(train_plus[["mention", "link", "hashtag",
                                     "city", "country", "keyword_bin"]])
dummy_cols = dummies.columns

comb = pd.concat([train["text"], dummies], axis=1)

# Split val and train
X_train, X_val, y_train, y_val = \
    train_test_split(comb, train["target"], 
                     test_size=0.2, random_state=42)


# 1st parameter is vocab size
"""
Parameters:
    vocab size - max number of words taken by tokenizer
    max_len - maximum length of the sequence
"""

vocab_size = 8000
max_len = 25

"""
Preprocessing Text
"""

# Text
text_train = preprocess_text(X_train["text"])
text_val = preprocess_text(X_val["text"])

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<oov>")
tokenizer.fit_on_texts(text_train)

padded_train, _ = do_padding(text_train, tokenizer, max_len, "post", "post")
padded_val, _ = do_padding(text_val, tokenizer, max_len, "post", "post")

"""
Model:
    - Text model
    - Others model (w outputs from text)
    
"""

# Text model
dropout_rate=0.3
text_model = Sequential([
    Embedding(vocab_size, 36, input_length = max_len),
    Bidirectional(LSTM(16,return_sequences=True,dropout=dropout_rate)),
    Bidirectional(LSTM(16,return_sequences=True,dropout=dropout_rate)),
    Bidirectional(LSTM(16,dropout=dropout_rate)),
    Dense(20, activation="relu"),
    Dense(1, activation="sigmoid")
])

text_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
text_model.summary()

history = text_model.fit(padded_train, y_train, epochs=2, 
                    validation_data=(padded_val, y_val), verbose=True)

text_pred = text_model.predict(padded_val)

accuracy_score(np.where(text_pred>0.5, 1, 0), y_val)

# Trying CNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

dropout_rate=0.3
text_model = Sequential([
    Embedding(vocab_size, 36, input_length = max_len),
    Conv1D(32, 3, padding='same', activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(20, activation="relu"),
    Dense(1, activation="sigmoid")
])

text_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
text_model.summary()

history = text_model.fit(padded_train, y_train, epochs=10, 
                    validation_data=(padded_val, y_val), verbose=True)



# Model other vars

from xgboost import XGBClassifier

add_model = XGBClassifier()
add_model.fit(X_train[dummy_cols], y_train)

other_model = Sequential([
    Dense(64, activation = "relu", input_shape = (len(dummy_cols), )),
    Dense(32, activation = "relu"),
    Dense(1, activation="sigmoid")    
])

other_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
other_model.summary()

history = other_model.fit(X_train[dummy_cols], y_train, epochs = 20, 
                validation_data=(X_val[dummy_cols], y_val))

other_pred = other_model.predict(X_val[dummy_cols])

accuracy_score(np.where(other_pred>0.5, 1, 0), y_val)

np.corrcoef(other_pred.reshape(-1), text_pred.reshape(-1))

# Concat Model

dropout_rate = 0.5

input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(len(dummy_cols),))

embedding_layer = Embedding(vocab_size, 36)(input_1)
lstm_1 = Bidirectional(LSTM(16, return_sequences=True, dropout=dropout_rate))(embedding_layer)
lstm_2 = Bidirectional(LSTM(16, return_sequences=True, dropout=dropout_rate))(lstm_1)
lstm_3 = Bidirectional(LSTM(16, dropout=dropout_rate))(lstm_2)
dense_1 = Dense(8, activation="relu")(lstm_3)

dense_2 = Dense(64, activation="relu")(input_2)
dropout_1 = Dropout(dropout_rate)(dense_2)
dense_3 = Dense(32, activation="relu")(dropout_1)
dropout_2 = Dropout(dropout_rate)(dense_3)
dense_4 = Dense(8, activation="relu")(dropout_2)

concat_layer = Concatenate()([dense_1, dense_4])
dropout_3 = Dropout(dropout_rate)(concat_layer)
# dense_4 = Dense(20, activation="relu")(concat_layer)
dense_4 = Dense(20, activation="relu")(dropout_3)
dropout_6 = Dropout(dropout_rate)(dense_4)
output = Dense(1, activation='sigmoid')(dropout_6)
# output = Dense(1, activation='sigmoid')(concat_layer)


model = Model(inputs=[input_1, input_2], outputs=output)

model.compile(loss='binary_crossentropy',
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              optimizer="adam",
              metrics=['accuracy'])
model.summary()

history = model.fit(x=[padded_train, X_train[dummy_cols]], y=y_train, 
                    epochs=5, verbose=1, 
                    validation_data=([padded_val, X_val[dummy_cols]], y_val))

# Ill go with this model. I'm still a bit annoyed that we can't increase 
# accuracy by combining features. I suppose that's likely due to overfitting.

# final_preds.py script fits model on full training set. 


text_pred_train = text_model.predict(padded_train)
other_pred_train = other_model.predict(X_train[dummy_cols])

train_x = np.concatenate((text_pred_train, other_pred_train ), axis=1)

text_pred_val = text_model.predict(padded_val)
other_pred_val = other_model.predict(X_val[dummy_cols])

val_x = np.concatenate((text_pred_val, other_pred_val), axis=1)

from sklearn.linear_model import LogisticRegression

stacked_mod = LogisticRegression()  
stacked_mod.fit(train_x, y_train)

comb_preds = stacked_mod.predict(val_x)

accuracy_score(y_val, comb_preds)