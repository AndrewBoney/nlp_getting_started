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
Researching use of additional columns
"""

# Split val and train
X_train, X_val, y_train, y_val = train_test_split(train[["keyword", "location"]], 
                                                  train["target"], 
                                                  test_size=0.2, random_state=1)


def do_padding(sentences, tokenizer, maxlen, padding, truncating):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    
    return padded, sequences

# Convert keywords to array of shape (, 3)
def process_keyword(keywords):
    keywords = keywords.str.lower()
    keywords = keywords.str.replace("%20", " ")
    keywords = np.where(keywords.isna(), "", keywords)
    
    return keywords

# Convert top 30 locations to sparse array of 1/0 flags.
# This isn't a great way to do this. It would be much better to preprocess
# this into e.g. [country, city] per row
def process_location(location):    
    location = location.to_frame()
    
    location_counts = location.value_counts().rename("counts").reset_index(). \
        rename(columns = {"index":"location"})
    location_counts["order"] = list(range(1, len(location_counts)+1))
    
    location = location.merge(location_counts, how="left", 
                   left_on = "location", right_on = "location")
    
    locations = np.select([location["order"] <= 30, location["order"].isna()],
              [location["location"], "other"],
              "missing")
    
    location_dummies = pd.get_dummies(locations)

    return location_dummies 

def get_location_cols(location):
    location_counts = location.value_counts()
    return location_counts.index[:30]

def location_dummies(location, cols):
    locations = np.select([location.isin(cols), location.isna()],
                          [location, "missing"],
                          "other")
    
    location_dummies = pd.get_dummies(locations)
    
    df_cols = location_dummies.columns
    
    for c in cols:
        col_exist = c in (df_cols)
        
        if ~col_exist:
            location_dummies.loc[:, c] = 0
        
    return location_dummies[list(cols) + ["missing", "other"]]
                              
# Keywords Preprocess
rk_train = process_keyword(X_train["keyword"])
rk_val = process_keyword(X_val["keyword"])

keyword_tokenizer = Tokenizer(num_words = 200, oov_token = "<oov>")
keyword_tokenizer.fit_on_texts(rk_train)

keywords_train, _ = do_padding(rk_train, keyword_tokenizer, 3, "post", "post")
keywords_val, _ = do_padding(rk_val, keyword_tokenizer, 3, "post", "post")

# Location
cols = get_location_cols(X_train["location"])
location_train = location_dummies(X_train["location"], cols)
location_val = location_dummies(X_val["location"], cols)

data_train = np.concatenate((keywords_train, location_train), axis=1)
data_val = np.concatenate((keywords_val, location_val), axis=1)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(20),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(data_train, y_train, epochs=100, 
                    validation_data=(data_val, y_val), verbose=True)
    
