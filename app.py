from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and LabelEncoders
with open('anaemia_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder_sex.pkl', 'rb') as le_sex_file:
    le_sex = pickle.load(le_sex_file)

with open('label_encoder_anaemic.pkl', 'rb') as le_anaemic_file:
    le_anaemic = pickle.load(le_anaemic_file)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        sex = request.form["Sex"]
        red = float(request.form["Red"])
        blue = float(request.form["Blue"])
        green = float(request.form["Green"])
        hb = float(request.form["Hb"])

        # Encode the 'Sex' field
        sex_encoded = le_sex.transform([sex])[0]

        # Make the prediction
        res = model.predict([[sex_encoded, red, blue, green, hb]])[0]

        # Convert the result to a human-readable form
        result = "Anemic" if res == 1 else "Not Anemic"

        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
