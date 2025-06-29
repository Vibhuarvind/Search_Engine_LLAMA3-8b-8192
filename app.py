import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search_functionality")

st.title("Langchain based Search Engine - Llama3-8b-8192")

"""
In this example, We're trying to use 'streamlitcallbackhandler' to display thoughts and actions of an agent in this interactive streamlit app.
[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent). 

"""

# sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter the Groq API key :",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[

       {"role":"assistant",
        "content":"Sup! I'm chatbot who can browse the web for you. How may I help you?"}
    ]

# we'll be traversing every msg

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#prompt creation
if prompt:=st.chat_input(placeholder="what is LLM observability?"):
    st.session_state.messages.append({"role":"user",
                                     "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key,
                   model="Llama3-8b-8192",
                   streaming=True)
    
    tools=[search,arxiv,wiki]


    # zero shot won't use previous responses context 
    search_agent = initialize_agent(tools,
                                llm,
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                handling_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),
                                         expand_new_thoughts=False)
        
        #getting response
        response = search_agent.run(st.session_state.messages,
                                    callbacks=[st_cb])
        
        st.session_state.messages.append({"role":"assistant",
                                          "content":response})
        
        st.write(response)
    