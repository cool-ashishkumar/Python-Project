#
from flask import Flask, render_template, request
import pickle
import streamlit as st


app = Flask(__name__)
# load the model
model = pickle.load(open('savedmodel.sav', 'rb'))


@app.route('/')
def home():
    return "Welcome All"

@app.route('/predict', methods= ['Get'])
def predict_iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):

    prediction = model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    print(prediction)
    return prediction


def main():
    st.title("Iris Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Prediction App </h2>
    </div> """
    st.markdown(html_temp, unsafe_allow_html=True)
    SepalLengthCm = st.text_input('Sepal Length', " ")
    SepalWidthCm = st.text_input('Sepal Width', " ")
    PetalLengthCm = st.text_input('Petal Length', " ")
    PetalWidthCm = st.text_input('Petal Width', " ")
    result = ""
    if st.button("Predict"):
        result=predict_iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    st.success('The Class Of Iris is {}'.format(result))


if __name__ =='__main__':
    main()