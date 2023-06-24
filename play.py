import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
import random
import pickle
import pyttsx3
import speech_recognition as sr

colorama.init()
from colorama import Fore, Style, Back

with open("intents.json") as file:
    data = json.load(file)


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(Fore.LIGHTBLUE_EX + "Speak:" + Style.RESET_ALL, end="")
        audio = r.listen(source)

    try:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        text = r.recognize_google(audio)
        print(text)
        return text.lower()
    except sr.UnknownValueError:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL + "Sorry, I didn't understand that.")
        return ""
    except sr.RequestError as e:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL + "Sorry, there was an issue with speech recognition. {0}".format(e))
        return ""


def chat():
    # Load trained model
    model = keras.models.load_model('chat_model')

    # Load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # Parameters
    max_len = 20

    # Initialize pyttsx3
    engine = pyttsx3.init()

    while True:
        inp = listen()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, response)
                engine.say(response)  # Convert the response to speech
                engine.runAndWait()  # Speak the response

    print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)


print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
