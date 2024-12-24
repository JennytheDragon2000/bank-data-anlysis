import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(filepath):
    # Load the dataset with proper quoting
    df = pd.read_csv(filepath, sep=';', quoting=1)  # quoting=1 for quote-minimal
    
    # Convert categorical variables to numerical
    le = LabelEncoder()
    # Updated categorical columns for bank-full.csv
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'poutcome', 'y']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle numeric columns for bank-full.csv
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 
                   'pdays', 'previous']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Split features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Create and train Neural Network model
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Main execution
def main():
    # Updated file path
    X, y = load_data('bank-full.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Neural Network
    print("Training Neural Network...")
    nn_model = create_nn_model(X_train.shape[1])
    history = nn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate models
    print("\nNeural Network Results:")
    nn_pred = (nn_model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, nn_pred))
    
    print("\nRandom Forest Results:")
    rf_pred = rf_model.predict(X_test)
    print(classification_report(y_test, rf_pred))
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot Neural Network training history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot feature importance from Random Forest
    plt.subplot(1, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': range(X_train.shape[1]),
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance (Random Forest)')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png')
    plt.close()

if __name__ == "__main__":
    ain() 