import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_excel('train.xlsx')

X = df.drop(columns='target', axis=1)
Y = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)


def predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction


def main():
    st.title("Logistic Regression Model Prediction App")

    st.sidebar.header("User Input")
    user_input = []
    for i in range(X.shape[1]):
        user_input.append(st.sidebar.number_input(f"Feature {i + 1}", value=0))

    if st.sidebar.button("Predict"):
        prediction = predict(user_input)
        st.write("prediction:", prediction)


if __name__ == "__main__":
    main()
