"""
train.py
---------
Trains a Deep Learning (LSTM) sentiment classifier FROM SCRATCH
on the IMDB movie reviews dataset (bundled with Keras, auto-downloads once).

Run this ONCE locally:
    python train.py

It saves two files after training:
    sentiment_model.h5   -> the trained DL model
    tokenizer.pkl        -> the tokenizer (needed to convert text -> numbers at inference time)

These two files are then loaded by app.py to serve live predictions.
"""

import pickle
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------- CONFIG ----------
VOCAB_SIZE = 10000      # only keep the top 10,000 most frequent words
MAX_LEN = 200           # pad/truncate every review to 200 words
EMBED_DIM = 64          # size of word embedding vectors
EPOCHS = 3              # increase to 5-10 for better accuracy (takes longer)
BATCH_SIZE = 64

print("Step 1/5: Loading IMDB dataset (25k train + 25k test reviews)...")
# IMDB dataset comes PRE-tokenized as integers already, but we also build
# our own Tokenizer below so the SAME preprocessing can be reused on
# raw user text coming from the web interface later.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print("Step 2/5: Padding sequences to fixed length...")
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

print("Step 3/5: Building the LSTM model architecture...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),  # 0 = negative, 1 = positive
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("Step 4/5: Training the model (this actually trains weights from scratch)...")
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

print("Step 5/5: Saving model + word index (tokenizer) for the web app...")
model.save("sentiment_model.h5")

# imdb.get_word_index() gives us word -> integer mapping used above.
# We save it so app.py can convert a user's raw sentence into the
# same integer format the model was trained on.
word_index = imdb.get_word_index()
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(word_index, f)

print("\nDONE. Files created: sentiment_model.h5, tokenizer.pkl")
print("Now run: python app.py")
