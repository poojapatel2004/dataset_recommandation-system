from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = joblib.load('dataset.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['query']
    input_vector = vectorizer.transform([user_input])

    distances, indices = model.kneighbors(input_vector)

    results = []
    for idx in indices[0]:
        dataset = {
            'Dataset Name': df.iloc[idx]['Dataset_name'],
            'Type of File': df.iloc[idx]['Type_of_file'],
            'Author': df.iloc[idx]['Author_name']
        }
        results.append(dataset)

    return render_template('results.html', query=user_input, recommendations=results)

if __name__ == '__main__':
    app.run(debug=True)
