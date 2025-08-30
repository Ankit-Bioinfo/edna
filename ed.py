import streamlit as st

# Title of the app
st.title("eDNA Biodiversity Assessment Tool")

# User input
uploaded_file = st.file_uploader("Upload your FASTA/FASTQ file", type=["fasta", "fastq"])

if uploaded_file:
    st.success("File uploaded successfully!")
    # You can add your analysis pipeline here
    st.write("Processing file:", uploaded_file.name)
else:
    st.info("Please upload an eDNA dataset to begin.")
