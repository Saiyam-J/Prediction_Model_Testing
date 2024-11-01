import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r'C:\Users\Public\Prediction Models\House_Price_India.csv')

# Create a binary target variable
median_price = df['Price'].median()
df['Price_Category'] = (df['Price'] > median_price).astype(int)

# Select features and target
features = [
    'number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 
    'number of floors', 'waterfront present', 'number of views', 'condition of the house', 
    'grade of the house', 'Area of the house(excluding basement)', 'Area of the basement',
    'Built Year', 'Renovation Year', 'Postal Code', 'Latitude', 'Longitude', 
    'living_area_renov', 'lot_area_renov', 'Number of schools nearby', 'Distance from the airport'
]
X = df[features]
y = df['Price_Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for Neural Networks and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Classification Report': class_report,
        'Training Time': training_time,
        'Prediction Time': prediction_time
    })

# Logistic Regression
evaluate_model('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)

# Random Forest
evaluate_model('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)

# Decision Tree
evaluate_model('Decision Tree', DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test)

# SVM
evaluate_model('SVM', SVC(random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)

# Neural Network
# Build Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Measure training time
start_time = time.time()
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
training_time = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
prediction_time = time.time() - start_time

# Evaluate the Neural Network model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

results.append({
    'Model': 'Neural Network',
    'Accuracy': accuracy,
    'Confusion Matrix': conf_matrix,
    'Classification Report': class_report,
    'Training Time': training_time,
    'Prediction Time': prediction_time
})

# Print results
for result in results:
    print(f"-------------------{result['Model']}-------------------")
    print(f"Accuracy: {result['Accuracy']}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print("Classification Report:")
    print(result['Classification Report'])
    print(f"Training Time: {result['Training Time']} seconds")
    print(f"Prediction Time: {result['Prediction Time']} seconds\n")

# Plotting Accuracy Comparison
models = [result['Model'] for result in results]
accuracies = [result['Accuracy'] for result in results]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.show()

# Plotting Training Time Comparison
training_times = [result['Training Time'] for result in results]

plt.figure(figsize=(10, 6))
plt.bar(models, training_times, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Model Training Time Comparison')
plt.xticks(rotation=45)
plt.show()

# Plotting Prediction Time Comparison
prediction_times = [result['Prediction Time'] for result in results]

plt.figure(figsize=(10, 6))
plt.bar(models, prediction_times, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Model')
plt.ylabel('Prediction Time (seconds)')
plt.title('Model Prediction Time Comparison')
plt.xticks(rotation=45)
plt.show() 
