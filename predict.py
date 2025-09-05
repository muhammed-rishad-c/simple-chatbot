import nltk 
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import warnings
import train_model

warnings.filterwarnings('ignore')

lemmatizer=WordNetLemmatizer()

intents=json.loads(open('intense.json').read())

data=pickle.load(open('chatbot.pkl','rb'))
vectorizer=data['vectorizer']
model=data['model']

def predict_intense(sentence):
    p=vectorizer.transform([sentence])
    predicted=model.predict(p)[0]
    return predicted

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag']==tag:
            return random.choice(intent['responses'])
        
    
    return "iam sorry i dont understand"