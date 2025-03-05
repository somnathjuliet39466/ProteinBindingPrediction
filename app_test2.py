from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pickle
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
import io
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

# Load the machine learning model
model = pickle.load(open('model1.pkl', 'rb'))

# Function to extract protein features
def extract_features(sequence):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_aa])
    if not cleaned_sequence:
        raise ValueError("Invalid sequence after cleaning")
    X = ProteinAnalysis(cleaned_sequence)
    properties = [
        X.molecular_weight(),
        X.aromaticity(),
        X.instability_index(),
        X.isoelectric_point(),
        X.secondary_structure_fraction()[0],  # Helix Fraction
        X.secondary_structure_fraction()[1],  # Turn Fraction
        X.secondary_structure_fraction()[2],  # Sheet Fraction
        X.molar_extinction_coefficient()[0],  # Extinction Coefficient (Reduced Cysteines)
        X.molar_extinction_coefficient()[1],  # Extinction Coefficient (Disulfide Bridges)
        X.gravy(),
        X.charge_at_pH(7.0)
    ]
    return properties

# Function to read user data from CSV
def read_user_data():
    file_path = 'user_data.csv'
    if not os.path.isfile(file_path):
        return []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

# Route to handle user information input
@app.route('/user_info', methods=['GET', 'POST'])
def user_info():
    if request.method == 'POST':
        # Save user data in the session
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        session['country'] = request.form['country']
        session['workplace'] = request.form['workplace']

        # Prepare the data to save
        user_data = {
            'Name': session['name'],
            'Email': session['email'],
            'Country': session['country'],
            'Workplace': session['workplace'],
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Date': datetime.now().strftime("%Y-%m-%d")
        }

        # Save to CSV file
        file_path = 'user_data.csv'
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['Name', 'Email', 'Country', 'Workplace', 'Time', 'Date'])
            if not file_exists:  # Add header only if the file does not exist
                writer.writeheader()
            writer.writerow(user_data)

        return redirect(url_for('predictor'))  # Redirect to the predictor page

    # Read user data to pass to the template
    user_data = read_user_data()
    user_count = len(user_data)
    user_locations = [user['Country'] for user in user_data]

    return render_template('user_info3.html', user_count=user_count, user_locations=user_locations)

# Home route that redirects to the user_info page
@app.route('/')
def home():
    return redirect(url_for('user_info'))  # Ensure redirection to user_info

# Route for predictor functionality
@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if not session.get('name'):  # Ensure user_info data is collected
        return redirect(url_for('user_info'))

    output = None
    error = None

    if request.method == 'POST':
        # Handle manual input, CSV upload, or FASTA upload
        if 'host_sequence' in request.form and 'pathogen_sequence' in request.form:
            try:
                host_sequence = request.form['host_sequence']
                pathogen_sequence = request.form['pathogen_sequence']
                host_features = extract_features(host_sequence)
                pathogen_features = extract_features(pathogen_sequence)
                input_features = [host_features + pathogen_features]
                prediction = model.predict(input_features)
                prob = 1 - model.predict_proba(input_features)[:, 1][0]
                output = f'Result: {"Non-Interacting" if prediction[0] == 1 else "Interacting"} with Probability: {prob:.4f}'
            except ValueError as e:
                error = str(e)

    return render_template('predictor.html', output=output, error=error)

if __name__ == '__main__':
    app.run(debug=True)