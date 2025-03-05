from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache

app = Flask(__name__)

# Dùng mô hình nhẹ hơn để tăng tốc
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache model để không load lại mỗi request
@lru_cache(maxsize=1)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model

# Hàm phân tích cảm xúc tối ưu
def analyze(text):
    tokenizer, model = get_model()
    
    # Chuyển input thành tensor, đưa về đúng device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1).cpu().numpy().tolist()[0]
    
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[scores.index(max(scores))]

    return {"text": text, "sentiment": sentiment, "scores": {"negative": scores[0], "neutral": scores[1], "positive": scores[2]}}

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = analyze(text)
    return jsonify(result)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
