import pickle
from flask import Flask, request, jsonify
import torch
import PyPDF2
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")


# Load pre-trained model and tokenizer from pickle files
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Function to process user query and generate answer
def generate_answer(question, documents_text):
    # Tokenize question and documents
    inputs = tokenizer(question, documents_text, return_tensors="pt", padding=True, truncation=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Find the start and end positions of the answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert token indices to actual tokens
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index + 1])

    return answer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Route to handle user questions
@app.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    pdf_file_path = "Final3.pdf"  # Update with the actual path to your PDF
    documents_text = extract_text_from_pdf(pdf_file_path)
    answer = generate_answer(question, documents_text)

    return jsonify({"answer": answer})

# Route to handle user questions
@app.route("/api/HrInd", methods=["POST"])
def ask_question_hr():
    data = request.get_json()
    question = data.get("question")
    pdf_file_path = "Induction.pdf"  # Update with the actual path to your PDF
    documents_text = extract_text_from_pdf(pdf_file_path)
    answer = generate_answer(question, documents_text)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
