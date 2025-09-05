import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import warnings
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',']

data_file=open('intense.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #print(pattern)
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        #print(f'\nwords : {words}')
        documents.append((w,intent['tag']))
        #print(f'\ndocuments : {documents}')
        
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        #print(classes)
        
words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

classes=sorted(list(set(classes)))

# print(words)
# print(documents)
# print(classes)


all_pattern=[patter for intent in intents['intents'] for patter in intent['patterns']]
all_tag=[intent['tag'] for intent in intents['intents'] for _ in intent['patterns']]

print(all_pattern)

print(f'all tags : {all_tag}')

vectorizer=TfidfVectorizer()
x_train=vectorizer.fit_transform(all_pattern)
y_train=all_tag

model=LinearSVC(max_iter=1000)
model.fit(x_train,y_train)



pickle.dump({'vectorizer':vectorizer,'model':model},open('chatbot.pkl','wb'))

print("training completed and model saved as chatbot.pkl file ")