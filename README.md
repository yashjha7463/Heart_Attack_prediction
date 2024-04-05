# Heart Attack Prediction App

This project involved collecting a heart attack dataset from Kaggle.
Performing exploratory data analysis (EDA) to understand the distribution of different features, correlations, and potentially identify important factor related to heart attacks. Testing the accuracy of various machine learning algorithms like random forest algorithm and also neural networks to develop a heart attack prediction model.
After fine-tuning the models, the most accurate one was selected (neural networks).
I deployed project using Streamlit to create a user-friendly interface for accessing the heart attack prediction tool.
The live project can be accessed at [ https://yash-haps.streamlit.app/].

## Getting Started

To run this application locally, follow these steps:

1. Clone this repository to your local machine using git clone.

2. Install the required dependencies: (`pip install -r requirements.txt`)

3. Download the heart disease dataset (`heart.csv`) and place it in the project directory.

4. Run the Streamlit app:
streamlit run app.py


5. Access the application in your web browser at `http://localhost:8501`.

## Usage

1. Use the sliders and dropdowns in the sidebar to input your age, sex, cholesterol level, fasting blood sugar, max heart rate, and resting blood pressure.

2. Click the "Predict" button to see the probability of having a heart attack based on the input data.

## Background Video

The application displays a background GIF while running. You can replace `background.gif` with your desired GIF file to customize the background.

## About the Model

The prediction model is built using a neural network trained on a heart disease dataset. It takes various features such as age, sex, cholesterol level, etc., as input and predicts the probability of having a heart attack.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or questions.












