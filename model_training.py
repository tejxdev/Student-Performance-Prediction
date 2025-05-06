import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode categorical columns
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['race/ethnicity'] = df['race/ethnicity'].astype('category').cat.codes
df['parental level of education'] = df['parental level of education'].astype('category').cat.codes
df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})
df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})

# Features and target
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
y = df['math score']  # we will predict math score

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")
