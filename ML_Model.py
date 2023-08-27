from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import PyPDF2
import pickle
# Load pre-trained QA model
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


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


# Path to the PDF file
pdf_file_path = "Final3.pdf"

# Extract text from the PDF
documents_text = extract_text_from_pdf(pdf_file_path)

# Example question
question = "How are working hours managed for employees assigned to client-sites?"

# Generate the answer
answer = generate_answer(question, documents_text)

print("Answer:", answer)

# Save the model and tokenizer as pickle files
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tokenizer.pkl", "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)