import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.api.layers import Input
from keras.api.datasets import imdb
import string

# required nltk modules
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

def load_n_split_data():
    (train_text, train_labels), (test_text, test_labels) = imdb.load_data(num_words=20000,
                                                                        skip_top=0,
                                                                        seed=113)
    # decode reviews into words
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    train_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in review]) for review in train_text]
    test_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in review]) for review in test_text]
    
    return train_text, train_labels, test_text, test_labels

def preprocess_data(dataset):
    # load stopwords
    stop_words = set(stopwords.words('english'))
    #intialise stemmer
    stemmer = PorterStemmer()
    # initialise lemmantizer
    lemmatizer = WordNetLemmatizer()
    
    # apply processing
    def preprocess_text(text):
        # standardise case
        text = text.lower()
        # remove puntuaction
        text = text.translate(str.maketrans('', '', string.punctuation))
        # remove stopwords and apply stemming and lemmatisation
        words = text.split()
        processed_words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]
        return ' '.join(processed_words)
    # apply over the entire dataset
    return [preprocess_text(text) for text in dataset]

def create_sentiment_model(vocab_size, embedding_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# load dataset
print("Loading Dataset: ")
train_text, train_labels, test_text, test_labels = load_n_split_data()
# print("Completed")
# preprocess data
print("Preprocessing training data: ")
prepro_train_text = preprocess_data(train_text)
# print("Completed")
print("Preprocessing test data: ")
prepro_test_text = preprocess_data(test_text)
# print("Completed")

# tokenize data
print("Tokenizing data: ")
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(prepro_train_text)

train_sequences = tokenizer.texts_to_sequences(prepro_train_text)
test_sequences = tokenizer.texts_to_sequences(prepro_test_text)

max_length = max(len(seq) for seq in train_sequences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# convert data to tensor format
train_data = train_padded
test_data =  test_padded

# print("Completed")

# build the model
vocab_size = 20000
embedding_dim = 100

input_shape = (1000,)
input_layer = Input(shape=input_shape)

# Create and train sentiment analysis model
vocab_size = 20000
embedding_dim = 100

sentiment_model = create_sentiment_model(vocab_size, embedding_dim)

sentiment_model.fit(train_data, train_labels, epochs=1, validation_split = 0.1, batch_size=64)

print(f"Test acc: {sentiment_model.evaluate(test_data, test_labels)[1]:.3f}")