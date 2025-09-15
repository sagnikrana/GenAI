import streamlit as st

# Page config
st.set_page_config(page_title="Chat Message Interface", page_icon="💬")

st.title("💬 Chat Message Interface Example")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response (dummy logic here)
    response = f"You said: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": response})

# import streamlit as st
# import openai
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama
# import os

# import os
# from dotenv import load_dotenv
# load_dotenv()

# ## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"

# ## Prompt Template
# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful massistant . Please  repsonse to the user queries"),
#         ("user","Question:{question}")
#     ]
# )

# def generate_response(question,llm,temperature,max_tokens):
#     llm=Ollama(model=llm)
#     output_parser=StrOutputParser()
#     chain=prompt|llm|output_parser
#     answer=chain.invoke({'question':question})
#     return answer

# ## #Title of the app
# st.title("Enhanced Q&A Chatbot With OpenAI")


# ## Select the OpenAI model
# llm=st.sidebar.selectbox("Select Open Source model",["gemma:2b"])

# ## Adjust response parameter
# temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
# max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# ## MAin interface for user input
# st.write("Go ahead and ask any question")
# user_input=st.text_input("You:")

# if user_input :
#     st.write("User i/p received")
#     response=generate_response(user_input,llm,temperature,max_tokens)
#     st.write(response)
# else:
#     st.write("Please provide the user input")