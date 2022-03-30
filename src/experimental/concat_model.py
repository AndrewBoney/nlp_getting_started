# Keywords Preprocess
rk_train = process_keyword(X_train["keyword"])
rk_val = process_keyword(X_val["keyword"])

keyword_tokenizer = Tokenizer(num_words = 200, oov_token = "<oov>")
keyword_tokenizer.fit_on_texts(rk_train)

keywords_train, _ = do_padding(rk_train, keyword_tokenizer, 3, "post", "post")
keywords_val, _ = do_padding(rk_val, keyword_tokenizer, 3, "post", "post")


cols = get_location_cols(X_train["location"])
location_train = location_dummies(X_train["location"], cols)
location_val = location_dummies(X_val["location"], cols)

# Combine
data_train = np.concatenate((keywords_train, location_train), axis=1)
data_val = np.concatenate((keywords_val, location_val), axis=1)

# input_1 = Input(shape=(max_len,))
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(data_train.shape[1],))

# embedding_layer = Embedding(vocab_size, 36, input_length=max_len+1)(input_1)
# lstm_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
lstm_1 = Bidirectional(LSTM(64, return_sequences=True))(input_1)
dropout_1 = Dropout(dropout_rate)(lstm_1)
lstm_2 = Bidirectional(LSTM(32))(dropout_1)

dense_1 = Dense(20, activation="relu")(input_2)
dropout_3 = Dropout(dropout_rate)(dense_1)
dense_2 = Dense(50, activation="relu")(dropout_3)

concat_layer = Concatenate()([lstm_2, dense_2])
dense_4 = Dense(20, activation="relu")(concat_layer)
dropout_6 = Dropout(dropout_rate)(dense_4)
output = Dense(1, activation='sigmoid')(dropout_6)
model = Model(inputs=[input_1, input_2], outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])
model.summary()

history = model.fit(x=[padded_train, data_train], y=y_train, 
                    epochs=20, verbose=1, 
                    validation_data=([padded_val, data_val], y_val))
