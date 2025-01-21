import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request,redirect, url_for
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load trained model and resources
try:
    model = load_model("chatbots_model2.h5")
    intents = json.loads(open("intents.json").read())
    words = pickle.load(open("words2.pkl", "rb"))
    classes = pickle.load(open("classes2.pkl", "rb"))
except Exception as e:
    print(f"Error loading files: {e}")  # Show error if loading fails
    exit(1)# Exit the program if any file loading fails

# Initialize Flask app
app = Flask(__name__)

# Routes for navigation
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Inbuilt username and password check (both set to 'admin')
        if username == "admin" and password == "admin":
            return redirect(url_for("chat"))  # Redirect to the chat page  # If correct, go to the chat page
        else:
            return "Incorrect username or password", 401  # Show error for incorrect login

    return render_template("login.html")

@app.route("/chat")
def chat():
    return render_template("chat.html") # Show the chat page where users can talk to the bot

@app.route("/about")
def about():
    return render_template("about.html")# Show an "About" page

# Chatbot response Handling
@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form.get("msg", "") # Get the message typed by the user
    if not msg:
        return "Message is missing.", 400   #If the message is empty, return an error

    try:
        if msg.lower().startswith('my name is') or msg.lower().startswith('hi my name is'):
            name = msg.split("is")[1].strip()# Extract name from the message
            ints = predict_class(msg, model) # Predict the intent using the chatbot model
            res1 = getResponse(ints, intents) # Get the response based on the predicted intent
            res = res1.replace("{n}", name)# Replace {n} with the user's name in the response
        else:
            ints = predict_class(msg, model) # Predict the intent from the message
            res = getResponse(ints, intents) # Get the response based on the predicted intent
    except Exception as e:
        res = f"Error processing message: {e}"    # If there's an error, return an error message

    return res      # Return the chatbot's response

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)   # Tokenize (split) the sentence into words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]   # Lemmatize the words
    return sentence_words   # Return the processed words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  # Clean up the sentence by breaking it into words
    bag = [0] * len(words)  #Create an empty list of zeros, one for each known word
    for s in sentence_words:     # For each word in the sentence, mark it as found in the bag (if it exists in the known words list)
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1        # Mark this word as found by setting the corresponding index to 1
                if show_details:
                    print(f"found in bag: {w}")     # Print a message if the word is found
    return np.array(bag)   # Return the bag as a numpy array (helps with calculations)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)  #convert the sentence to a bag of words
    if len(p.shape) == 1:
        p = np.expand_dims(p, axis=0)   # Reshape if needed
    res = model.predict(p)[0]   # Use the model to predict the intent based on the bag of words
    ERROR_THRESHOLD = 0.25     # Minimum probability threshold for prediction
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Get results with probability > threshold
    results.sort(key=lambda x: x[1], reverse=True)   # Sort results by highest probability
    if not results:# If no intent is predicted, return a default response
        return [{"intent": "default", "probability": "0"}]
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]    # Return the predicted intent


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]  # Get the predicted intent tag
    for intent in intents_json["intents"]:  # Loop through the intents
        if intent["tag"] == tag:   # If the intent matches the predicted intent
            return random.choice(intent["responses"])    # Pick a random response for that intent
    return "Sorry, I didn't understand that."   # Return a default message if no intent matches

if __name__ == "__main__":  # This makes sure the app runs only if this script is executed directly
    app.run(port=5006, debug=True)     # Start the Flask web app on port 5006, with debug mode on



# **Language and Libraries:**
# - **Python**: We are using Python because it's simple and very powerful for tasks like making a chatbot. It has many libraries that help us handle text, numbers, machine learning, and web development all in one language.
# - **Flask**: A Python library used to create web applications (websites). It helps us make a web page where users can talk to the chatbot.
# - **Keras/TensorFlow**: Libraries used for creating and working with machine learning models. The chatbot uses this to understand what users say and respond accurately.
# - **NumPy**: A library used for handling large amounts of data (arrays) and performing mathematical operations. It's essential for the chatbot to process the data efficiently.
# - **NLTK**: A library used to understand and manipulate natural language. We need it to break sentences into words and prepare those words for the chatbot.
# - **Pickle**: Used for saving and loading data such as models, words, and responses in a simple file format.
