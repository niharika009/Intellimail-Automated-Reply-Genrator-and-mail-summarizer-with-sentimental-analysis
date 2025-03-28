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
            st.write(f"**Sentiment:** {email_data['sentiment']['label']} (Score: {email_data['sentiment']['score']:.2f})")
            st.write("**Summary:**")
            st.text_area("Email Summary", email_data['summary'], height=100)
            st.write("**Suggested Reply:**")
            st.text_area("AI-Generated Reply", email_data['reply'], height=100)

            # Send AI Reply Button (Uses a callback to avoid page reset)
            st.button("Send AI Reply ✉️", on_click=send_reply)

    # Display reply status if available
    if st.session_state.reply_status:
        st.success(st.session_state.reply_status) if "✅" in st.session_state.reply_status else st.error(st.session_state.reply_status)

if __name__ == "__main__":
    main()
