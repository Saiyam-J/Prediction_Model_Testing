import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

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

# Feature scaling (important for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Measure training time
start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
training_time = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
prediction_time = time.time() - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"Training Time: {training_time} seconds")
print(f"Prediction Time: {prediction_time} seconds")
