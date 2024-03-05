import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model
model = LogisticRegression(random_state=42)

# Function to preprocess and scale new data for prediction
def preprocess_data(new_data, scaler, X_train_columns):
    # One-hot encoding for new data
    new_data = pd.get_dummies(new_data, columns=['sex'], drop_first=True)

    # Ensure new data columns match the training data columns
    missing_cols = set(X_train_columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0

    # Reorder columns to match the order in X_train
    new_data = new_data[X_train_columns]

    # Feature scaling for new data
    new_data_scaled = scaler.transform(new_data)

    return new_data_scaled

# Load the dataset
df = pd.read_csv("heart.csv")  # Replace with the actual path to your dataset

# Drop rows with missing values in the target variable 'output'
df = df.dropna(subset=['output'])

# Select features and target variable
selected_features = ['age', 'sex', 'chol', 'fbs', 'trtbps', 'thalachh']
X = df[selected_features]
y = df['output']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model.fit(X_scaled, y)

# Get the columns of X_train for later use
X_train_columns = X.columns

# User input for prediction
age = 40
sex = 1  # Male
chol = 200
fbs = 0  # False
thalachh = 120
trtbps = 120

# Create a DataFrame for new data
new_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chol': [chol],
    'fbs': [fbs],
    'thalachh': [thalachh],
    'trtbps': [trtbps],
})

# Preprocess and scale the new data
new_data_scaled = preprocess_data(new_data, scaler, X_train_columns)

# Predict the probability of having a heart attack
probability = model.predict_proba(new_data_scaled)[:, 1]

# Output the prediction
if probability > 0.5:
    print(f"Probability of having a Heart Attack: {probability[0]*100:.2f}%")
    print("You are at heavy risk☠☠!See your doctor immediately")
else:
    print("Probability of having a Heart Attack: {probability[0]*100:.2f}%")
    print("Nothing is serious!Maintain a healthy diet and exercise")
