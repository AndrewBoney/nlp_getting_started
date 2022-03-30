import numpy as np 
import pandas as pd 

from tensorflow.keras import Input
from keras.layers.core import Dropout, Dense
from keras.layers import LSTM, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

from tensorflow.keras.preprocessing.text import Tokenizer

from src.utils import *
from model import (do_padding,get_extra,preprocess_text,convert_cities,convert_countries)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Processing extra vars
keyword_bins = pd.read_csv("data/keyword_bins.csv", dtype={"keyword":str, 
                                                           "keyword_bin":str})
locations = pd.read_csv("data/locations.csv")

def process_extra_vars(df, keyword_bins=keyword_bins, locations=locations):
    df_plus = df.merge(keyword_bins, how="left", on = "keyword"). \
        merge(locations, how="left", on="location")
    
    df_plus.loc[df_plus["keyword_bin"].isna(), "keyword_bin"] = "missing"
    
    df_plus = convert_cities(df_plus)
    df_plus = convert_countries(df_plus)
    df_plus = get_extra(df_plus)
    
    dummies = pd.get_dummies(df_plus[["mention", "link", "hashtag",
                                         "city", "country", "keyword_bin"]])
    dummy_cols = dummies.columns
    
    return dummies, dummy_cols

train_dummies, train_dummy_cols = process_extra_vars(train)
test_dummies, test_dummy_cols = process_extra_vars(test)

train_dummy_cols.difference(test_dummy_cols)

# Given that these countries don't exist in test, and we're building a new 
# model, I'm going to drop these

train_dummies.drop(["country_south africa","country_spain"],axis=1,inplace=True)

# ensuring the same order
test_dummies = test_dummies[train_dummies.columns]

# Processing text

vocab_size = 8000
max_len = 25

"""
Preprocessing Text
"""

# Text
text_train = preprocess_text(train["text"])
text_test = preprocess_text(test["text"])

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<oov>")
tokenizer.fit_on_texts(text_train)

padded_train, _ = do_padding(text_train, tokenizer, max_len, "post", "post")
padded_test, _ = do_padding(text_test, tokenizer, max_len, "post", "post")

"""
Model
Concatenated tensorflow model
"""

dropout_rate = 0.5

input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(len(train_dummies.columns),))

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
dense_4 = Dense(20, activation="relu")(dropout_3)
dropout_6 = Dropout(dropout_rate)(dense_4)
output = Dense(1, activation='sigmoid')(dropout_6)

model = Model(inputs=[input_1, input_2], outputs=output)

model.compile(loss='binary_crossentropy',
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              optimizer="adam",
              metrics=['accuracy'])
model.summary()

history = model.fit(x=[padded_train, train_dummies], y=train["target"], 
                    epochs=5, verbose=1)

preds = model.predict([padded_test, test_dummies])
preds_target = np.where(preds>0.5, 1, 0).reshape(-1)

submission = pd.DataFrame({"id":test["id"],
                           "target":preds_target})

submission.to_csv("submission.csv", index=False)

