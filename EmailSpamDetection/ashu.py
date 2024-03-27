# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
#
# ps = PorterStemmer()
#
#
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
# model = pickle.load(open('emu.sav','rb'))
#
# st.title("Email/SMS Spam Classifier")
#
# input_sms = st.text_area("Enter the message")
#
# if st.button('Predict'):
#
#     # 1. preprocess
#     vector_input = transform_text(input_sms)
#     # 2. vectorize
#
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
ps = PorterStemmer()

# Load the trained model
model = pickle.load(open('emu.sav', 'rb'))

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features= 2)

@st.cache
def transform_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    # Remove non-alphanumeric characters and stem words
    processed_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]
    return ' '.join(processed_tokens)

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input text
    vector_input = transform_text(input_sms)
    # Transform the input text using the fitted vectorizer
    vector_input = vectorizer.fit_transform([vector_input])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

