import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

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
