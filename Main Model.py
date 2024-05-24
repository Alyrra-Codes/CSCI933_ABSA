import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Load IMDB dataset
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Decode IMDB data back to text
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([index_word.get(i - 3, '?') for i in encoded_review])

x_train_text = [decode_review(review) for review in x_train]
x_test_text = [decode_review(review) for review in x_test]

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

preprocessed_x_train_text = preprocess_data(x_train_text)
preprocessed_x_test_text = preprocess_data(x_test_text)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(preprocessed_x_train_text)
x_train_seq = tokenizer.texts_to_sequences(preprocessed_x_train_text)
x_test_seq = tokenizer.texts_to_sequences(preprocessed_x_test_text)

max_len = 250
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

# Define the basic sentiment analysis model
def create_basic_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

basic_model = create_basic_model()

# Train the basic sentiment analysis model
history_basic = basic_model.fit(x_train_pad, y_train, validation_data=(x_test_pad, y_test), epochs=3, batch_size=64)

# Plotting training history for basic sentiment analysis
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history_basic)

# Define some common aspects related to movies
aspects = [
    "Plot", "Acting", "Direction", "Screenplay", "Music", "Visuals",
    "Character development", "Dialogues", "Cinematography", "Special effects",
    "Soundtrack", "Editing", "Pacing", "Storyline", "Setting", "Costumes",
    "Makeup", "Casting", "Lighting", "Production design", "Performances",
    "Script", "Score", "Humor", "Suspense", "Drama", "Action sequences",
    "Romance", "Climax", "Opening scene", "Ending", "Background score",
    "Theme", "Atmosphere", "Emotional impact", "Character arcs", "Subplots",
    "Voice acting", "Stunts", "Visual effects", "Art direction", "Camera work",
    "Plot twists", "Genre elements", "Symbolism", "Tone", "Dialogue delivery",
    "Narration", "Adaptation", "Source material", "Scene transitions",
    "Character chemistry", "Tension", "Conflict", "Resolution", "Morals",
    "Message", "Inspiration", "Creativity", "Originality", "Believability",
    "Realism", "Fantastical elements", "World-building", "Character relationships",
    "Villain", "Protagonist", "Antagonist", "Side characters", "Plot holes",
    "Consistency", "Continuity", "Real-life relevance", "Symbolic elements",
    "Allegory", "Metaphors", "Cinematic style", "Film techniques", "Directorial vision",
    "Auteur influence", "Franchise impact", "Sequel setup", "Prequel elements",
    "Flashbacks", "Dream sequences", "Real-time sequences", "Time jumps",
    "Non-linear narrative", "Subtext", "Irony", "Satire", "Parody", "Homage",
    "Tribute", "Cultural references", "Societal impact", "Political commentary",
    "Historical accuracy", "Futuristic elements", "Speculative fiction"
]

# Extract aspects from reviews
def extract_aspects(review, aspects):
    found_aspects = []
    for aspect in aspects:
        if aspect.lower() in review.lower():
            found_aspects.append(aspect)
    return found_aspects

# Function to create aspect-based labels for training
def create_aspect_labels(reviews, aspects):
    labels = []
    for review in reviews:
        extracted_aspects = extract_aspects(review, aspects)
        label = [1 if aspect in extracted_aspects else 0 for aspect in aspects]
        labels.append(label)
    return np.array(labels)

# Create aspect-based labels for training and testing datasets
aspect_labels_train = create_aspect_labels(x_train_text, aspects)
aspect_labels_test = create_aspect_labels(x_test_text, aspects)

# Define the ABSA model
class ABSAModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, max_len, num_aspects):
        super(ABSAModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, input_length=max_len)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(num_aspects, activation='sigmoid')  # Use sigmoid for multi-label classification

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        attention_output = self.attention([x, x])
        x = tf.reduce_mean(attention_output, axis=1)
        return self.dense(x)

vocab_size = 10000
embed_size = 128
num_aspects = len(aspects)

absa_model = ABSAModel(vocab_size, embed_size, max_len, num_aspects)
absa_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for multi-label classification

# Train the aspect-based sentiment model
history_absa = absa_model.fit(x_train_pad, aspect_labels_train, validation_split=0.2, epochs=3, batch_size=64)

# Plot training history for ABSA
plot_history(history_absa)

# Evaluate the ABSA model
absa_model.evaluate(x_test_pad, aspect_labels_test)
