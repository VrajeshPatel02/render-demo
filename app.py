from flask import Flask, request, render_template
import pickle
import os
import re

def preprocess_text(text):
    """Clean and preprocess item description text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the models and encoders from the .pkl file
with open('model.pkl', 'rb') as f:
    loaded_models_and_encoders_pkl = pickle.load(f)

loaded_model_category_pkl = loaded_models_and_encoders_pkl['model_category']
loaded_model_item_pkl = loaded_models_and_encoders_pkl['model_item']
loaded_model_sub_item_pkl = loaded_models_and_encoders_pkl['model_sub_item']
loaded_model_size_pkl = loaded_models_and_encoders_pkl['model_size']
loaded_le_category_pkl = loaded_models_and_encoders_pkl['le_category']
loaded_le_item_pkl = loaded_models_and_encoders_pkl['le_item']
loaded_le_sub_item_pkl = loaded_models_and_encoders_pkl['le_sub_item']
loaded_le_size_pkl = loaded_models_and_encoders_pkl['le_size']
loaded_tfidf_vectorizer_pkl = loaded_models_and_encoders_pkl['tfidf_vectorizer']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    item_description = request.form.get('item_description', '')
    
    clean_item_description = preprocess_text(item_description)
    X_new = loaded_tfidf_vectorizer_pkl.transform([clean_item_description])
    
    predicted_category = loaded_le_category_pkl.inverse_transform(loaded_model_category_pkl.predict(X_new))[0]
    predicted_item = loaded_le_item_pkl.inverse_transform(loaded_model_item_pkl.predict(X_new))[0]
    predicted_sub_item = loaded_le_sub_item_pkl.inverse_transform(loaded_model_sub_item_pkl.predict(X_new))[0]
    predicted_size = loaded_le_size_pkl.inverse_transform(loaded_model_size_pkl.predict(X_new))[0]
    
    output_array = [predicted_category, predicted_item, predicted_sub_item, predicted_size]
    prediction_text = f"Category: {predicted_category}, Item: {predicted_item}, Sub Item: {predicted_sub_item}, Size: {predicted_size}"
    
    return render_template('index.html', prediction_text=prediction_text, output_array=output_array)

if __name__ == "__main__":
    app.run(debug=True)
