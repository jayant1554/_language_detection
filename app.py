from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and CountVectorizer
model = pickle.load(open('lang_MNb_model.pkl', 'rb'))
cv = pickle.load(open('lang_countvectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if data and 'text' in data:
        text = data['text']
        text_vectorized = cv.transform([text])
        lang = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max()
        return jsonify({'language': lang, 'confidence': confidence})
    return jsonify({'error': 'No text provided'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0")
