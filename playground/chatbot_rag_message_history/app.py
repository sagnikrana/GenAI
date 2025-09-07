import streamlit as st
import time

from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.notebook import remove_newlines
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()


# Interface for user to submit the PDF

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Read the PDF
documents = None
vector_db = None
cleaned_docs = None

embeddings=(OllamaEmbeddings(model="gemma:2b"))  ##by default it ues llama2. gemma:2b is downloaded in local pc
llm = Ollama(model="gemma:2b")


if uploaded_file :
    st.write("treating the uploaded file")
    # Step 2: Save the uploaded file temporarily (PyPDFLoader can work with file paths)
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 3: Load the PDF file using PyPDFLoader from the saved path
    loader = PyPDFLoader("temp_uploaded.pdf")
    documents = loader.load()

# Get the embeddings and store it into the Chroma DB interface

embeddings=(OllamaEmbeddings(model="gemma:2b"))  ##by default it ues llama2


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

start = time.time()

if documents:
    st.write("Splitting the documents and persisting ")
    splits = text_splitter.split_documents(documents)
    st.write('Have the splits')
    vector_db = Chroma.from_documents(splits,embeddings, persist_directory = '..\stored_data\chroma_db')
    st.write("Persisted")
    # Process 1
    end = time.time()
    st.write(f"Process 1 took {end - start} seconds")
        
    cleaned_docs = [
        Document(page_content=remove_newlines(doc.page_content), metadata=doc.metadata)
        for doc in splits
    ]
    st.write("Cleaned docs are ready.")
    # Write code to summarize the document using the Ollama Embedding
    prompt=ChatPromptTemplate.from_template(
        """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    """
    )

    document_chain=create_stuff_documents_chain(llm,prompt)
    response = document_chain.invoke({
    "input" : "Please summarize in short the document that's provided to you",
        "context" :cleaned_docs
    })
    st.write(f"Here is the summary of the file you provided : {response}")

    # Tell user that it's ready
    if vector_db:
        st.write("the data is ready !!")
        query = st.text_input('Please enter your question about the document')
        if len(query) > 0 : 
            st.write(f"Got your question : Your question is  : {query}")
            docs_from_query = vector_db.similarity_search(query)
            st.write("Following are the results from the similarity search : ")
            for i, doc in enumerate(docs_from_query):
                st.write(f"Result {i+1}")
                st.write(f"Content:\n{doc.page_content.strip().replace("\n", "")}")
                st.write(f"Metadata: {doc.metadata}")
                st.write("-" * 40)


# Provide users option to select the temperature and other objectives. 


# Mention how much of chat history we need to provide.


# Handle exceptions and prepare for shipping