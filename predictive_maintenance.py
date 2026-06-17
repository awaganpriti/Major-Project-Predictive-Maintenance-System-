"""
Project Title: Industrial Predictive Maintenance System using Machine Learning
Author: Electronics & Telecommunication Engineering Student
Description: This system monitors industrial sensor logs (vibration, temperature, pressure)
             and uses a Random Forest Classifier to predict equipment failures before they happen.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# STEP 1: SIMULATE SENSOR DATA ACQUISITION
# ==========================================
def generate_sensor_data(num_samples=1000):
    """
    Simulates real-world sensor streams from an industrial machine.
    Features: Temperature (°C), Vibration (mm/s), Pressure (psi)
    Target: 0 = Normal Operation, 1 = Maintenance Required (Failure Risk)
    """
    np.random.seed(42)
    
    # Simulating normal operational ranges
    temperature = np.random.normal(loc=65, scale=10, size=num_samples)  # Avg 65°C
    vibration = np.random.normal(loc=4.2, scale=1.5, size=num_samples)  # Avg 4.2 mm/s
    pressure = np.random.normal(loc=45, scale=5, size=num_samples)     # Avg 45 psi
    
    # Initialize all labels as Normal (0)
    failure_label = np.zeros(num_samples, dtype=int)
    
    # Inject synthetic failure rules (simulating real machine anomalies)
    for i in range(num_samples):
        # Rule 1: High Temperature combined with High Vibration indicates bearing friction
        if temperature[i] > 82 and vibration[i] > 6.5:
            failure_label[i] = 1
        # Rule 2: Sudden drop in pressure with high temperature indicates structural leak/overheating
        elif pressure[i] < 34 and temperature[i] > 78:
            failure_label[i] = 1
        # Rule 3: Extreme isolated anomalies
        elif vibration[i] > 8.5:
            failure_label[i] = 1
            
    # Compile features into a structured Dataframe
    data = pd.DataFrame({
        'Temperature_C': temperature,
        'Vibration_mms': vibration,
        'Pressure_psi': pressure,
        'Maintenance_Required': failure_label
    })
    return data

# ==========================================
# STEP 2: MODEL TRAINING PIPELINE
# ==========================================
def train_predictive_model(df):
    """
    Splits data, trains a Random Forest Classifier, and evaluates performance.
    """
    # Extract Features (X) and Labels (y)
    X = df[['Temperature_C', 'Vibration_mms', 'Pressure_psi']]
    y = df['Maintenance_Required']
    
    # Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[INFO] Training Machine Learning Model (Random Forest Classifier)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[SUCCESS] Model trained successfully with Accuracy: {accuracy * 100:.2f}%")
    
    # Display full classification matrix to show precision/recall metrics
    print("\n--- Model Classification Report ---")
    print(classification_report(y_test, predictions))
    
    return model

# ==========================================
# STEP 3: REAL-TIME ANOMALY INFERENCE EVENT
# ==========================================
def live_telemetry_check(model, test_reading, machine_id="MC_098"):
    """
    Simulates checking a single incoming telemetry packet from a live machine edge-node.
    """
    # Convert input array into a structured dataframe match feature names
    reading_df = pd.DataFrame([test_reading], columns=['Temperature_C', 'Vibration_mms', 'Pressure_psi'])
    
    prediction = model.predict(reading_df)[0]
    probability = model.predict_proba(reading_df)[0][1] # Probability of failure (class 1)
    
    print(f"\n--- Live Diagnostics for Machine ID: {machine_id} ---")
    print(f"Current Metrics -> Temp: {test_reading[0]}°C | Vibration: {test_reading[1]} mm/s | Pressure: {test_reading[2]} psi")
    
    if prediction == 1:
        print(f" [ALERT] CRITICAL VALUE DETECTED. Failure Probability: {probability*100:.1f}%")
        print(" [ACTION REQUIRED] Scheduling Automated Preventive Maintenance Dispatch immediately.")
    else:
        print(f" [STATUS] System Normal. Failure Probability: {probability*100:.1f}%. No anomaly flags.")

# ==========================================
# MAIN INITIALIZATION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("====================================================")
    print("     INDUSTRIAL PREDICTIVE MAINTENANCE DASHBOARD    ")
    print("====================================================")
    
    # 1. Fetch data
    dataset = generate_sensor_data(num_samples=1200)
    print(f"[DATA] Successfully acquired telemetry records. Shape: {dataset.shape}")
    print(dataset.head()) # Shows the first 5 log entries
    
    # 2. Train model 
    trained_clf = train_predictive_model(dataset)
    
    # 3. Simulate operational checks
    # Simulation Case A: Normal operational telemetry parameters
    normal_machine_telemetry = [62.4, 3.8, 46.1]
    live_telemetry_check(trained_clf, normal_machine_telemetry)
    
    # Simulation Case B: Unstable anomalous telemetry parameters
    failing_machine_telemetry = [86.5, 7.1, 42.0]
    live_telemetry_check(trained_clf, failing_machine_telemetry)
    
    print("\n====================================================")