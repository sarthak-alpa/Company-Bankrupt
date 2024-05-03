# Company-Bankrupt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("C:\Sarthak Prompt\data.csv")
print(df.head())

# Display the first few rows of the DataFrame
print("DataFrame Head:")
print(df.head())

# Split the data into features (X) and target variable (y)
X = df.drop('Bankrupt?', axis=1)  # Features (all columns except 'Bankrupt?')
y = df['Bankrupt?']                # Target variable ('Bankrupt?' column)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Display classification report (includes precision, recall, F1-score, etc.)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
