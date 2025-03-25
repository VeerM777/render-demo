from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
       
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))

       
        input_data = np.array([cgpa, iq, profile_score]).reshape(1, 3)
        print(f"Input data: {input_data}")  

        
        result = model.predict(input_data)
        print(f"Prediction result: {result}")  

        
        if result[0] == 1:
            result = 'Placed'
        else:
            result = 'Not Placed'

        return render_template('index1.html', result=result)

    except Exception as e:
        
        print(f"Error: {e}")
        return render_template('index1.html', result="Error: Something went wrong.")

if __name__ == '__main__':
    app.run(debug=True)
