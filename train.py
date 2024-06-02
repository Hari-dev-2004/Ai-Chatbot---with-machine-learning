import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import joblib

def load_data(file_path='chatbot_data.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    training_sentences = []
    training_labels = []
    responses = {}

    for intent in data:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['intent'])
        responses[intent['intent']] = intent['responses']

    return training_sentences, training_labels, responses

def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def train_model(file_path='chatbot_data.json', model_path='chatbot_model.h5', encoder_path='label_encoder.pkl', vectorizer_path='vectorizer.pkl'):
    data = load_data(file_path)
    training_sentences, training_labels, responses = preprocess_data(data)
    
    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(training_labels)
    joblib.dump(label_encoder, encoder_path)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_sentences).toarray()
    y = np.array(training_labels)

    model = create_model(X.shape[1], len(label_encoder.classes_))

    model.fit(X, y, epochs=200, batch_size=5, verbose=1)
    model.save(model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == '__main__':
    train_model()
