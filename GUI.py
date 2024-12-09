import streamlit as st,requests,os
from PIL import Image
with st.sidebar:
    icon=Image.open("icon.png")
    st.image(icon,width=140,caption="Classifying Messages with Precision: Detect Spam and Keep Your Inbox Clean!")
    st.title("About the Program (for the Streamlit GUI)👇")
    st.write("""
Program Description:\n
This application is a user-friendly GUI for a spam detection system, designed to identify and classify messages as spam or non-spam. Using advanced machine learning techniques, the system analyzes text input, processes it through a feature extraction pipeline, and predicts whether the message is legitimate or spam. The application aims to streamline spam detection tasks with an intuitive interface, making it accessible for both technical and non-technical users.""")
    st.divider()
    st.title("GROUP MEMBERS ON THIS PROJECT : ")
    st.write("1.Bruce-Arhin Shadrach, 20061815")
    st.write("2.Tirth Doshi, 200609650")
    st.write("3.Chandrika Ghale, 200575692")
    st.write("4.Derick Appiah, 200584981")



icon=Image.open("spam.png")
st.image(icon,width=300)
st.header(":red[SPAM DETECTION SYSTEM]",divider='orange')
st.write(":blue[Classifying Messages with Precision: Detect Spam and Keep Your Inbox Clean!]")

st.divider()
if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for messages in st.session_state.messages:
    with st.chat_message(messages["Role"]):
        st.markdown(messages['info'])

if text:=st.chat_input("Enter Text here !!!"):
    
        try:
            url="http://127.0.0.1:8500/Model_prediction"
            with st.spinner():
                proced=requests.post(url,json={'txt':text})
                st.session_state.messages.append({'Role':"user",'info':text})
                if proced.status_code==200:
                    with st.chat_message('assistant'):
                        st.markdown(proced.text)
                st.session_state.messages.append({'Role':"Spam",'info':proced.text})        
        except Exception as ex:
            st.warning(f":red[Server Error occured, Please check your internet connection and try again]", icon="⚠️")    