# import streamlit as st
# import requests

# # Initialize session state to store reply status
# if "reply_status" not in st.session_state:
#     st.session_state.reply_status = None

# def fetch_email():
#     response = requests.get("http://127.0.0.1:8000/process-email")
#     if response.status_code == 200:
#         return response.json()
#     return None

# def send_reply():
#     response = requests.post("http://127.0.0.1:8000/send-reply")
    
#     if response.status_code == 200:
#         st.session_state.reply_status = "Reply sent successfully! ✅"
#     else:
#         st.session_state.reply_status = "Failed to send reply. ❌"

# def main():
#     st.title("📧 Inteli-Mail: Smart Email Assistant")
    
#     if st.button("Fetch Latest Email"):
#         email_data = fetch_email()
        
#         if email_data:
#             st.subheader("📨 Email Details")
#             st.write(f"**Subject:** {email_data['subject']}")
#             st.write(f"**From:** {email_data['from']}")
#             st.write("**Body:**")
#             st.text_area("Email Content", email_data['body'], height=200)

#             st.subheader("🧠 Processed Results")
#             st.write(f"**Sentiment:** {email_data['sentiment']['label']} (Score: {email_data['sentiment']['score']:.2f})")
#             st.write("**Summary:**")
#             st.text_area("Email Summary", email_data['summary'], height=100)
#             st.write("**Suggested Reply:**")
#             st.text_area("AI-Generated Reply", email_data['reply'], height=100)

#             # Send AI Reply Button (Uses a callback to avoid page reset)
#             st.button("Send AI Reply ✉️", on_click=send_reply)

#     # Display reply status if available
#     if st.session_state.reply_status:
#         st.success(st.session_state.reply_status) if "✅" in st.session_state.reply_status else st.error(st.session_state.reply_status)

# if __name__ == "__main__":
#     main()
import streamlit as st
import requests

# Initialize session state to store reply status
if "reply_status" not in st.session_state:
    st.session_state.reply_status = None

def fetch_email():
    response = requests.get("http://127.0.0.1:8000/process-email")
    if response.status_code == 200:
        return response.json()
    return None

def send_reply():
    response = requests.post("http://127.0.0.1:8000/send-reply")
    
    if response.status_code == 200:
        st.session_state.reply_status = "Reply sent successfully! ✅"
    else:
        st.session_state.reply_status = "Failed to send reply. ❌"

def main():
    st.title("📧 Inteli-Mail: Smart Email Assistant")
    
    if st.button("Fetch Latest Email"):
        email_data = fetch_email()
        
        if email_data:
            st.subheader("📨 Email Details")
            st.write(f"**Subject:** {email_data['subject']}")
            st.write(f"**From:** {email_data['from']}")
            st.write("**Body:**")
            st.text_area("Email Content", email_data['body'], height=200)

            st.subheader("🧠 Processed Results")

            # Display Sentiment
            st.write(f"**Sentiment:** {email_data['sentiment']['label']} (Score: {email_data['sentiment']['score']:.2f})")

            # Display Summary & ROUGE Scores
            st.write("**Summary:**")
            st.text_area("Email Summary", email_data['summary'], height=100)

            st.write("### 📊 ROUGE Scores (Summarization Quality)")
            rouge_scores = email_data.get("rouge_scores", {})

            if rouge_scores:
                st.write(f"**ROUGE-1** (F1: {rouge_scores['ROUGE-1']['f1-score']:.2f}, P: {rouge_scores['ROUGE-1']['precision']:.2f}, R: {rouge_scores['ROUGE-1']['recall']:.2f})")
                st.write(f"**ROUGE-2** (F1: {rouge_scores['ROUGE-2']['f1-score']:.2f}, P: {rouge_scores['ROUGE-2']['precision']:.2f}, R: {rouge_scores['ROUGE-2']['recall']:.2f})")
                st.write(f"**ROUGE-L** (F1: {rouge_scores['ROUGE-L']['f1-score']:.2f}, P: {rouge_scores['ROUGE-L']['precision']:.2f}, R: {rouge_scores['ROUGE-L']['recall']:.2f})")
            else:
                st.write("ROUGE scores not available.")

            # Display AI-Generated Reply
            st.write("**Suggested Reply:**")
            st.text_area("AI-Generated Reply", email_data['reply'], height=100)

            # Send AI Reply Button (Uses a callback to avoid page reset)
            st.button("Send AI Reply ✉️", on_click=send_reply)

    # Display reply status if available
    if st.session_state.reply_status:
        st.success(st.session_state.reply_status) if "✅" in st.session_state.reply_status else st.error(st.session_state.reply_status)

if __name__ == "__main__":
    main()
