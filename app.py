import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

tfidf = pickle.load(open('./model/vectorizer.pkl','rb'))
model = pickle.load(open('./model/model.pkl','rb'))

def transform_message(mess):
    mess = mess.lower()
    mess = nltk.word_tokenize(mess)
    
    newMess = []
    for i in mess:
        if i.isalnum():
            newMess.append(i)
            
    mess = newMess[:]
    newMess.clear()
    for i in mess:
        if i not in stopwords.words('english'):
            newMess.append(i)
            
    mess = newMess[:]
    newMess.clear()
    for i in mess:
        newMess.append(ps.stem(i))
            
    return ' '.join(newMess)

st.title('SMS Spam Classifier')

inputSMS = st.text_area('Enter the message')

if st.button('Predict'):
    # 1. preprocess
    transformedSMS = transform_message(inputSMS)
    # 2. vectorize
    vectorInput = tfidf.transform([transformedSMS])
    # 3. predict
    result = model.predict(vectorInput)[0]
    # 4. display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')