
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Model names
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
REPLY_GENERATION_MODEL_NAME = "google/flan-t5-large"

# Load models with correct architecture
def load_sentiment_model():
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    return model, tokenizer

def load_summarization_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    return model, tokenizer

def load_reply_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(REPLY_GENERATION_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(REPLY_GENERATION_MODEL_NAME)
    return model, tokenizer

# Load all models
sentiment_model, sentiment_tokenizer = load_sentiment_model()
summarization_model, summarization_tokenizer = load_summarization_model()
reply_model, reply_tokenizer = load_reply_model()

# Sentiment Analysis
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    
    label = outputs.logits.argmax().item()

    # Map label to human-readable sentiment
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return {"label": sentiment_labels.get(label, "Unknown"), "score": float(outputs.logits.softmax(dim=1).max())}

# Summarization
def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = summarization_model.generate(**inputs, max_length=100, min_length=30, num_beams=5)
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Reply Generation (Ensuring it's unique and not a summary)
def generate_reply(email_body, summary_text):
    prompt = (
        "You are a professional email assistant. Your task is to generate a polite, well-structured, and concise email reply.\n"
        "Guidelines:\n"
        "1️⃣ Start with a warm greeting (e.g., Dear [Sender's Name]),\n"
        "2️⃣ Acknowledge the sender's message genuinely,\n"
        "3️⃣ Address key concerns or questions from the email,\n"
        "4️⃣ Avoid unnecessary repetition or copying the summary,\n"
        "5️⃣ Maintain a professional tone,\n"
        "6️⃣ Conclude with a warm and appropriate closing statement.\n\n"
        f"Original Email:\n{email_body}\n\n"
        f"Email Summary (for context, do NOT copy this directly):\n{summary_text}\n\n"
        "Now, generate a unique, structured, and professional reply."
    )

    inputs = reply_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    reply_ids = reply_model.generate(
        **inputs, 
        max_length=200,  
        min_length=50,  
        num_beams=5,  
        do_sample=True,  
        temperature=0.4,  
        top_p=0.9  
    )

    reply_text = reply_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Deduplicate sentences in the reply
    unique_sentences = []
    for sentence in reply_text.split(". "):
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    reply_text = ". ".join(unique_sentences)

    return reply_text.strip()

# Process Email Content
def process_email_content(email_body):
    summary = summarize_text(email_body)
    return {
        "sentiment": analyze_sentiment(email_body),
        "summary": summary,
        "reply": generate_reply(email_body, summary)
    }
