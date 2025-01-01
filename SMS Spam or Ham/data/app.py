import streamlit as st

import joblib

# Load the trained model
model = joblib.load('spam_or_ham.pkl')
vectorizer = joblib.load("vectorizer.pkl")


st.markdown(
    """
    <style>
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    </style>
    <div class="centered-content">
        <h1>Spam Detection App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# Input box for the user to enter a message
user_input = st.text_area( " ", placeholder="Enter your message")

# Button to classify the message
if st.button("Classify"):
    if user_input:
        # Make a prediction
        
        prediction = model.predict(vectorizer.transform([user_input]))

        # Display the result
        if prediction[0] == 1:
            st.write("Prediction: **Spam**")
        else:
            st.write("Prediction: **Ham**")
    else:
        st.write("Please enter a message to classify.")
