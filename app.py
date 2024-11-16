# import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("models/model.pkl", "rb"))
except FileNotFoundError:
    model = None
    print("Error: 'model.pkl' file not found. Please ensure the model file is in the correct directory.")

# Kolom fitur yang digunakan dalam model
features = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness',
            'Neuroticism', 'sleep_time', 'wake_time', 'sleep_duration', 'PSQI_score',
            'call_duration', 'num_calls', 'num_sms', 'screen_on_time', 'skin_conductance',
            'accelerometer', 'mobility_radius', 'mobility_distance']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan input dari form
    input_data = []
    
    # Mendapatkan nilai input untuk setiap fitur
    for feature in features:
        try:
            value = request.form[feature]
            # Cek apakah inputnya kosong
            if value == '':
                raise ValueError(f"{feature} cannot be empty.")
            # Menambahkan input ke dalam list sebagai float
            input_data.append(float(value))
        except ValueError as e:
            # Jika ada kesalahan input, tampilkan pesan kesalahan
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

    # Normalisasi input menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform([input_data])  # Menormalkan input

    # Prediksi menggunakan model
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        # Jika terjadi kesalahan saat prediksi, tampilkan pesan kesalahan
        return render_template('index.html', prediction_text=f"Prediction Error: {str(e)}")

    # Mapping hasil prediksi ke label
    stress_levels = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
    result = stress_levels.get(prediction[0], "Unknown")

    return render_template('index.html', prediction_text=f'Predicted Stress Level: {result}')

if __name__ == '__main__':
    app.run(debug=True)
