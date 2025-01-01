from flask import Flask, render_template, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from torch.nn import functional as F

app = Flask(__name__)

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("results")  # Use the saved model path
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define route for home page (HTML form)
@app.route('/')
def home():
    return render_template('index.html')  # The HTML file where input is taken

# Define route for prediction (API endpoint)
@app.route('/predict', methods=['POST'])
def predict():
    # Get review text from the frontend
    review_text = request.form['review_text']  # The input from the form

    # Tokenize the input
    inputs = tokenizer(review_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    print(predicted_class)

    prediction_label = ""
    # Map the predicted class to 'positive' or 'negative'
    if predicted_class == 1:
        prediction_label = "positive"
    else:
        prediction_label = "negative"

    print(prediction_label)   

    # Return the result as a JSON response
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
