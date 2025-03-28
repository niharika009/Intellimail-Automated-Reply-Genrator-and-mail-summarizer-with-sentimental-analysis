from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gmail_api import get_latest_email
from nlp_processing import process_email_content
from gmail_api import send_email_reply
import uvicorn

app = FastAPI()

# Enable CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Email Processing API Running"}

# @app.get("/process-email")
# def process_email():
#     email = get_latest_email()
#     if not email:
#         return {"error": "No emails found"}

#     # Call NLP processing to analyze email
#     processed_results = process_email_content(email["body"])

#     return {
#         "subject": email["subject"] or "No Subject",
#         "from": email["from"] or "Unknown Sender",
#         "body": email["body"] or "No Email Body Found",
#         "sentiment": processed_results["sentiment"],
#         "summary": processed_results["summary"],
#         "reply": processed_results["reply"]
#     }

@app.get("/process-email")
def process_email():
    email = get_latest_email()
    if not email:
        raise HTTPException(status_code=404, detail="No emails found")

    # Call NLP processing to analyze email
    processed_results = process_email_content(email.get("body", ""))

    return {
        "subject": email.get("subject", "No Subject"),
        "from": email.get("from", "Unknown Sender"),
        "body": email.get("body", "No Email Body Found"),
        "sentiment": processed_results["sentiment"],
        "summary": processed_results["summary"],
        "rouge_scores": processed_results["rouge_scores"],  # Display ROUGE scores
        "reply": processed_results["reply"]
    }

    
@app.post("/send-reply")
def send_reply():
    email = get_latest_email()
    if not email:
        return {"error": "No emails found"}

    processed_results = process_email_content(email["body"])
    reply_text = processed_results["reply"]

    # Send the email reply
    success = send_email_reply(email["from"], reply_text)
    
    if success:
        return {"message": "Reply sent successfully"}
    else:
        return {"error": "Failed to send reply"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
