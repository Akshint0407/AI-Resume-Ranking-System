import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
# Add a background image
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: white;  /* Change text color to white for better visibility */
        }}
        .popup {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            background-color: rgba(0, 0, 0, 0.9); /* Black background */
            border-radius: 10px;
            padding: 20px;
            z-index: 1000;
            color: white; /* White text */
        }}
        .popup h4 {{
            margin-top: 0;
        }}
        .button-container {{
            position: absolute;
            top: 10px;
            right: 10px;
        }}
        .button {{
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent button */
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            cursor: pointer;
            margin-left: 5px; /* Space between buttons */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Replace with your new static image URL
add_background("https://img.freepik.com/free-vector/gradient-black-backgrounds-with-golden-frames_23-2149148385.jpg?semt=ais_hybrid")

st.title("AI Resume Ranking System")
# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
