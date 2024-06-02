from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

def load_data(file_path='chatbot_data.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    training_sentences = []
    training_labels = []
    labels = []
    responses = {}

    for intent in data:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['intent'])
        responses[intent['intent']] = intent['responses']

    return training_sentences, training_labels, responses

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    model = load_model('chatbot_model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    x = vectorizer.transform([text]).toarray()
    prediction = model.predict(x)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
    data = load_data()
    responses = preprocess_data(data)[2]
    bot_response = np.random.choice(responses[predicted_label])

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
