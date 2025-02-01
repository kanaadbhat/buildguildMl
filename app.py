import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from flask import Flask, request, jsonify
import joblib

# Generate Synthetic Construction Project Data
def generate_project_data(num_samples=10):
    np.random.seed(42)
    data = {
        'project_size': np.random.uniform(1000, 100000, num_samples),
        'complexity': np.random.randint(1, 10, num_samples),
        'team_experience': np.random.uniform(1, 20, num_samples),
        'location': np.random.choice(['urban', 'rural', 'suburban'], num_samples),
        'material_cost': np.random.uniform(10000, 500000, num_samples),
        'labor_cost': np.random.uniform(50000, 1000000, num_samples),
        'project_duration': np.random.uniform(3, 36, num_samples),
        'delay_risk': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'maintenance_cost': np.random.uniform(5000, 200000, num_samples),
        'worker_productivity': np.random.uniform(50, 100, num_samples)
    }
    return pd.DataFrame(data)

# Project Cost Estimation (Linear Regression)
def project_cost_estimation(data):
    X = data[['project_size', 'complexity', 'team_experience']]
    y = data['material_cost'] + data['labor_cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, 'models/cost_estimation_model.pkl')
    joblib.dump(scaler, 'models/cost_scaler.pkl')
    
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    return model, scaler, mse

# Project Delay Prediction (Random Forest Classification)
def project_delay_prediction(data):
    X = data[['project_size', 'complexity', 'team_experience']]
    y = data['delay_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/delay_prediction_model.pkl')
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Equipment Maintenance Prediction (Support Vector Regression)
def equipment_maintenance_prediction(data):
    X = data[['project_size', 'complexity', 'team_experience']]
    y = data['maintenance_cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, 'models/maintenance_prediction_model.pkl')
    joblib.dump(scaler, 'models/maintenance_scaler.pkl')
    
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    return model, scaler, mse

# Worker Productivity Analysis (K-Means Clustering)
def worker_productivity_clustering(data):
    X = data[['project_size', 'team_experience', 'worker_productivity']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    joblib.dump(kmeans, 'models/productivity_cluster_model.pkl')
    return kmeans

# Flask API Setup
app = Flask(__name__)

@app.route('/predict/cost', methods=['POST'])
def predict_cost():
    data = request.json
    model = joblib.load('models/cost_estimation_model.pkl')
    scaler = joblib.load('models/cost_scaler.pkl')
    
    input_data = pd.DataFrame(data, index=[0])
    scaled_data = scaler.transform(input_data[['project_size', 'complexity', 'team_experience']])
    prediction = model.predict(scaled_data)
    
    return jsonify({'estimated_cost': prediction[0]})

@app.route('/predict/delay', methods=['POST'])
def predict_delay():
    data = request.json
    model = joblib.load('models/delay_prediction_model.pkl')
    
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data[['project_size', 'complexity', 'team_experience']])
    
    return jsonify({'delay_risk': int(prediction[0])})

@app.route('/predict/maintenance', methods=['POST'])
def predict_maintenance():
    data = request.json
    model = joblib.load('models/maintenance_prediction_model.pkl')
    scaler = joblib.load('models/maintenance_scaler.pkl')

    input_data = pd.DataFrame(data, index=[0])
    scaled_data = scaler.transform(input_data[['project_size', 'complexity', 'team_experience']])
    prediction = model.predict(scaled_data)

    return jsonify({'predicted_maintenance_cost': prediction[0]})

@app.route('/predict/productivity', methods=['POST'])
def predict_productivity():
    data = request.json
    model = joblib.load('models/productivity_cluster_model.pkl')

    input_data = pd.DataFrame(data, index=[0])
    cluster = model.predict(input_data[['project_size', 'team_experience', 'worker_productivity']])

    return jsonify({'productivity_cluster': int(cluster[0])})

# Main Execution
if __name__ == '__main__':
    project_data = generate_project_data()

    # Train models
    cost_model, cost_scaler, cost_mse = project_cost_estimation(project_data)
    delay_model, delay_accuracy = project_delay_prediction(project_data)
    maintenance_model, maintenance_scaler, maintenance_mse = equipment_maintenance_prediction(project_data)
    productivity_model = worker_productivity_clustering(project_data)

    # Start Flask app
    app.run(debug=True)
