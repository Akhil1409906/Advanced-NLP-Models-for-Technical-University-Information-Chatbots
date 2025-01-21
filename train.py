import random
from tensorflow.keras.optimizers import Adam  # Using Adam optimizer instead of SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping  # For early stopping

# Download necessary NLTK resources
nltk.download('omw-1.4')  # For WordNet synsets
nltk.download("punkt")     # For tokenizing words
nltk.download("wordnet")   # For WordNet lemmatization

lemmatizer = WordNetLemmatizer()

# Initialize data structures
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Tokenize words and build the dataset
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))

        # Add to classes if not already present
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize, lower, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes for later use
pickle.dump(words, open("words2.pkl", "wb"))
pickle.dump(classes, open("classes2.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # Lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to numpy arrays
random.shuffle(training)
train_x = [sample[0] for sample in training]
train_y = [sample[1] for sample in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# Build the neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model with Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# EarlyStopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor="accuracy", patience=50, restore_best_weights=True)

# Train the model with more epochs and early stopping
model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=1, callbacks=[early_stopping])

# Save the model
model.save("chatbots_model2.h5")
print("Model created and saved.")
