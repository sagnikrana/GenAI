import streamlit as st
import time

from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Interface for user to submit the PDF

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Read the PDF
documents = None
vector_db = None

if uploaded_file :
    st.write("treating the uploaded file")
    # Step 2: Save the uploaded file temporarily (PyPDFLoader can work with file paths)
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 3: Load the PDF file using PyPDFLoader from the saved path
    loader = PyPDFLoader("temp_uploaded.pdf")
    documents = loader.load()
    st.write("here")

# Get the embeddings and store it into the Chroma DB interface

embeddings=(OllamaEmbeddings(model="gemma:2b"))  ##by default it ues llama2


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

start = time.time()

if documents:
    st.write("Splitting the documents and persisting ")
    splits = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(splits,embeddings, persist_directory = '..\stored_data\chroma_db')
    st.write("Persisted")
    # Process 1
    end = time.time()
    print(f"Process 1 took {end - start} seconds")



# Write code to summarize the document using the Ollama Embedding 

# Tell user that it's ready
if vector_db:
    st.write("the data is ready !!")
    name = st.text_input('Please enter your question about the document')


# User interface to submit question 

# Take the info and do a similarity search

# Validate the output

# return the output to the user.






# Turn the user search answer box to a query based interface


# Provide users option to select the temperatire and other objectives. 


# Mention how much of chat history we need to provide.


# Handle exceptions and prepare for shipping