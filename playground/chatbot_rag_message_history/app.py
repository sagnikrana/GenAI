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
    st.write("here")

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
   # "input" : "What is the biggest fault in the boeing aircrafts as per the article?",
   "input" : "What happened with the 737 max aircrafts?",
    "context" :cleaned_docs
})

messages=[
    SystemMessage(content="Answer about the problems that boeing has faced in the recent times"),
    HumanMessage(content="What challenges Boeing 737 max has faced in recent times?")
]

parser=StrOutputParser()
# parser.invoke(result)

chain=llm|parser

# Invoke contains the list about the messages. Could be i/p or many forms of messagyes, system, human, AI etc..
chain.invoke(messages)



# Tell user that it's ready
if vector_db:
    st.write("the data is ready !!")
    name = st.text_input('Please enter your question about the document')

# Take the question and do a similarity search

# Validate the output

# return the output to the user.





# Turn the user search answer box to a query based interface


# Provide users option to select the temperature and other objectives. 


# Mention how much of chat history we need to provide.


# Handle exceptions and prepare for shipping