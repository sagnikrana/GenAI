import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.notebook import remove_newlines
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Page config
st.set_page_config(page_title="Document QA Chat", layout="wide")
st.title("📄 Document Q&A Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

if "documents" not in st.session_state:
    st.session_state.documents = None

if "llm" not in st.session_state:
    # Initialize your LLM here
    from langchain_openai import ChatOpenAI

    # st.session_state.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    st.session_state.llm = Ollama(model="gemma:2b")

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                # Load and process PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)

                cleaned_docs = [
                    Document(
                        page_content=remove_newlines(doc.page_content),
                        metadata=doc.metadata
                    )
                    for doc in splits
                ]

                st.session_state.documents = cleaned_docs
                st.session_state.messages = []  # Reset chat history
                st.session_state.store = {}  # Reset message store
                st.success("✅ PDF processed successfully!")

            finally:
                os.unlink(tmp_path)


# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# Main chat interface
if st.session_state.documents is None:
    st.info("👈 Please upload a PDF file to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create prompt template
                    prompt = ChatPromptTemplate.from_template(
                        """
                        Answer the following question based only on the provided context:
                        <context>
                        {context}
                        </context>
                        
                        Question: {input}
                        Chat History : {chat_history}
                        """
                    )

                    # Create document chain
                    document_chain = create_stuff_documents_chain(
                        st.session_state.llm,
                        prompt
                    )

                    # Wrap with message history
                    document_chain_with_hist = RunnableWithMessageHistory(
                        document_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                    )

                    # Invoke chain
                    response = document_chain_with_hist.invoke(
                        {
                            "input": user_input,
                            "context": st.session_state.documents
                        },
                        config={"configurable": {"session_id": "main_session"}},
                    )

                    st.markdown(response)

                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Upload a PDF and ask questions about its content. The bot will maintain conversation history.")