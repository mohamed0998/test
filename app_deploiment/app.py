import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(request.form.values())
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='La reponse a votre requete {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


