from load import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import nltk

data = pd.read_csv('./data/data.csv')

for index, row in data.iterrows():
    if row['Action'] == 'Others':
        data = data.drop([index])

vocab = []
for index, row in data.iterrows():
    tokens = nltk.word_tokenize(row['Query'])
    for i in tokens:
        if not i in vocab:
            vocab.append(i)

vocab.append('UNK')
vocab.append('PAD')

messages = {
    'get_event_fees': 'The registration fees is Rs.499. Both include Speaker sessions and Hack-a-thon.',
    'is_refundable': 'Sorry, the event fee is non-refundable.',
    'get_registration_date': 'The registrations have already begun. Register now, and follow our facebook page to stay updated!',
    'get_payment_method': 'On registering at the website , you will be redirected to the payment portal.',
    'get_prizes': 'Haha, that\'s a surprise! ;)',
    'get_discounts': 'Sorry, there are no discounts yet!',
    'greet': 'Hello there!I am chatty, ask me anything about this event. ',
    'show_schedule': 'The itinerary has not been finalized yet. We will get back to you soon!',
    'get_event_date': 'This event is happening on 17th and 18th of March, 2019.',
    'get_event_time': 'The timings are from 9 am to 6pm',
    'show_accomodation': 'Sorry, we do not have information regarding accommodation. You may contact us via our Facebook page :)',
    'show_speakers': 'As of now, we have mentors and speakers from Microsoft and Google.',
    'speaker_details_extra': 'As of now, we have mentors and speakers from Microsoft and Google.',
    'show_food_arrangements': 'Haha, you seem to be hungry. But, sorry, we have not yet finalized the food arrangements.',
    'get_distance': 'Hmmm, check google maps?',
    'get_location': 'The event is happening in VIT Vellore. venue-Technology Tower Gallery2.',
    'show_contact_info': 'You may contact us on our Facebook page :)',
    'about_chatbot': 'I am Chatty, a smart assistant that can answer all your queries regarding our current Event, Evento. What would you like to know?'
}

n_words = len(vocab)
actions = list(data['Action'].unique())
n_actions = len(actions)

action_index_1 = {}
action_index_2 = {}

for i, v in enumerate(actions):
    action_index_1[i] = v
    action_index_2[v] = i


def get_index_matrix(sentence):
    matrix = []
    w = nltk.word_tokenize(sentence)
    for i in w:
        if i in vocab:
            matrix.append(vocab.index(i))
        else:
            matrix.append(vocab.index('UNK'))
    x = pad_sequences(maxlen=18, sequences=[matrix], padding="post", value=vocab.index('PAD'))
    return x

def get_prediction(query):
    a = nltk.word_tokenize(query)
    a = [i.lower() for i in a]
    a = [i for i in a if i.isalpha()]
    sentence = ' '.join(a)
    x = get_index_matrix(sentence)
    prediction = model.predict([x])[0]
    ans = np.argmax(prediction)
    score = round(max(prediction) * 100, 2)
    return action_index_1[ans], score




global model, graph
model, graph = init()


import tkinter
from tkinter import *
from PIL import ImageTk,Image
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    action, score = get_prediction(msg)
    if msg!= '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="white", font=("Verdana", 12 ))
        res = format(messages[action])
        ChatLog.insert(END, "chatBot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("Chatbot support")


my_img=ImageTk.PhotoImage(Image.open("botimg.png"))
my_label=Label(image=my_img)

my_label.pack()


base.geometry("750x500")
base.resizable(width=TRUE, height=TRUE)

ChatLog = Text(base, bd=0, bg="black",fg="white", height="8", width="80", font="Arial",)
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#990145", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

EntryBox = Text(base, bd=0, bg="black",fg="white",width="29", height="5", font="Arial")

scrollbar.place(x=450,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=445)
EntryBox.place(x=135, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()

























