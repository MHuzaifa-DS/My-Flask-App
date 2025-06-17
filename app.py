from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_text():
    input_text = request.form['text']
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)

    # Plagiarism Percentage Estimation
    words = input_text.split()
    plagiarized_words = [word for word in words if model.predict(tfidf_vectorizer.transform([word]))[0] == 1]
    percentage = (len(plagiarized_words) / len(words)) * 100 if words else 0

    # Highlight
    highlighted_text = " ".join([f"<span class='highlight'>{word}</span>" if word in plagiarized_words else word for word in words])
    
    if result[0] == 1:
        message = f"Plagiarism Detected – {percentage:.2f}% of the content appears copied."
    else:
        message = "✅ No Plagiarism Detected"

    return render_template('index.html', result=message, highlighted=highlighted_text)


if __name__ == '__main__':
    app.run(debug=True)
