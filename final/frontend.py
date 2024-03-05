import streamlit as st
import requests

# Streamlit UI for uploading video and displaying severity report
st.title('Stuttering Severity Analyzer')

uploaded_file = st.file_uploader("Upload video file", type=['mp4'])

if uploaded_file is not None:
    st.write('File uploaded successfully!')
    if st.button('Analyze'):
        # Send audio file for analysis to Flask backend
        files = {'file': uploaded_file}
        response = requests.post('http://localhost:5000/upload', files=files)
        if response.status_code == 200:
            result = response.json()
            st.subheader('Severity Report')
            st.write(f'Stuttering severity: {result["severity"]}')
            st.write(f'Severity label: {result["label"]}')
            st.write(f'Description: {result["description"]}')
        else:
            st.write('Error analyzing file. Please try again.')
