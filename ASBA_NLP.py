import os
import pathlib
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.tokenize import word_tokenize

# Download and extract the IMDB dataset
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!rm -r aclImdb/train/unsup

# Download NLTK data
nltk.download('punkt')

# Prepare directories for validation data
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

for category in ("neg", "pos"):
    os.makedirs(val_dir / category, exist_ok=True)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname, val_dir / category / fname)

# Load data using text_dataset_from_directory
batch_size = 32
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

# Define TextVectorization layer
max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Make a text-only dataset (without labels), then call `adapt`
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Define some aspect keywords
aspect_keywords = {
    "acting": ["actor", "actress", "acting", "performance", "role", "cast"],
    "story": ["story", "plot", "narrative", "script", "screenplay"],
    "visuals": ["visual", "cinematography", "scene", "shot", "special effects"],
    "sound": ["sound", "music", "score", "audio"],
    "direction": ["director", "direction", "filmmaker", "filmmaking"]
}

# Vectorize the text and extract labels
def extract_text_and_labels(dataset, vectorize_layer):
    texts = []
    labels = []
    for text_batch, label_batch in dataset:
        vectorized_texts = vectorize_layer(text_batch).numpy()
        texts.append(vectorized_texts)
        labels.append(label_batch.numpy())
    return np.concatenate(texts), np.concatenate(labels)

train_texts, train_labels = extract_text_and_labels(train_ds, vectorize_layer)
val_texts, val_labels = extract_text_and_labels(val_ds, vectorize_layer)
test_texts, test_labels = extract_text_and_labels(test_ds, vectorize_layer)

# Tokenize and vectorize the reviews to extract aspects
def extract_aspects(reviews, aspect_keywords, sequence_length, vectorize_layer):
    vocab = vectorize_layer.get_vocabulary()
    word_to_index = {word: index for index, word in enumerate(vocab)}
    
    aspect_vectors = []
    for review in reviews:
        aspect_vector = np.zeros(sequence_length)
        for i, token_index in enumerate(review[:sequence_length]):
            token = vocab[token_index] if token_index < len(vocab) else ""
            for aspect, keywords in aspect_keywords.items():
                if token in keywords:
                    aspect_vector[i] = 1
        aspect_vectors.append(aspect_vector)
    return np.array(aspect_vectors)

train_aspects = extract_aspects(train_texts, aspect_keywords, sequence_length, vectorize_layer)
val_aspects = extract_aspects(val_texts, aspect_keywords, sequence_length, vectorize_layer)
test_aspects = extract_aspects(test_texts, aspect_keywords, sequence_length, vectorize_layer)

# Define the ABSA model
input_text = keras.Input(shape=(sequence_length,), dtype='int32')
input_aspect = keras.Input(shape=(sequence_length,), dtype='float32')

embedding_layer = layers.Embedding(input_dim=max_features + 1, output_dim=embedding_dim)
embedded_text = embedding_layer(input_text)

lstm_out = layers.LSTM(128, return_sequences=True)(embedded_text)

# Reshape aspect input
aspect_dense = layers.Dense(128, activation='relu')(input_aspect)

attention_out = layers.Attention()([lstm_out, aspect_dense])
avg_pool = layers.GlobalAveragePooling1D()(attention_out)

output = layers.Dense(1, activation='sigmoid')(avg_pool)

absa_model = keras.Model(inputs=[input_text, input_aspect], outputs=output)
absa_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ABSA model
epochs = 10
history = absa_model.fit(
    [train_texts, train_aspects],
    train_labels,
    validation_data=([val_texts, val_aspects], val_labels),
    epochs=epochs,
    batch_size=batch_size
)

# Evaluate the ABSA model
loss, accuracy = absa_model.evaluate([test_texts, test_aspects], test_labels)
print(f"ABSA Test Accuracy: {accuracy:.2f}")

