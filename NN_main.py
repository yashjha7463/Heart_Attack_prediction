import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

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
new_data_scaled = preprocess_data(new_data, scaler, X.columns)

# Predict the probability of having a heart attack
probability = model.predict(new_data_scaled)[0]

# Output the prediction
if probability > 0.5:
    print(f"Probability of having a Heart Attack: {probability[0]*100:.2f}%")
    print("You are at heavy risk☠☠! See your doctor immediately")
else:
    print(f"Probability of having a Heart Attack: {probability[0]*100:.2f}%")
    print("Nothing is serious! Maintain a healthy diet and exercise")
