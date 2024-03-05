import streamlit as st
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

# Streamlit app
def main():
    st.title("Heart Attack Prediction App")
    # Display the background image
    st.image('heaart.gif')

    # Sidebar with user input
    st.sidebar.header("User Input")
    age = st.sidebar.slider("Age", min_value=20, max_value=80, value=40)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    chol = st.sidebar.slider("Cholesterol", min_value=100, max_value=400, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
    thalachh = st.sidebar.slider("Max Heart Rate", min_value=60, max_value=220, value=120)
    trtbps = st.sidebar.slider("Resting Blood Pressure", min_value=90, max_value=200, value=120)

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

    if st.button("Predict"):
        # Predict the probability of having a heart attack
        probability = model.predict(new_data_scaled)[0]

        st.success(f"Probability of having a Heart Attack: {probability[0]*100:.2f}%")

        if probability > 0.5:
            st.success("You are at heavy risk☠☠! See your doctor immediately")
        else:
            st.success("Nothing is serious! Maintain a healthy diet and exercise")

if __name__ == "__main__":
    main()
