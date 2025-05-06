from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/student_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [int(request.form['gender']),
            int(request.form['race']),
            int(request.form['education']),
            int(request.form['lunch']),
            int(request.form['prep'])]

    prediction = model.predict([np.array(data)])
    return render_template('index.html', prediction_text=f"Predicted Math Score: {prediction[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
