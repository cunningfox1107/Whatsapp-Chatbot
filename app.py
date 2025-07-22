import streamlit as st
import re
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Optional
from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
os.environ['OPENAI_API_KEY'] =''
os.environ['OPENAI_API_KEY']= os.getenv("OPENAI_API_KEY")
os.environ['SERPAPI_API_KEY'] ='ed5c2efa4940cf2e4667d38efef90eb32aacf1da6f0f3f3c260d4a4e7dd3e6a8'
os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")

loader = TextLoader("cleaned_name_chat.txt", encoding='utf-8')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ".", " "])
split_text = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(split_text, embeddings)
retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={"k": 10, "lambda_mult": 0.7})

llm = ChatOpenAI(model='gpt-4o', temperature=0.3)
output_parser = StrOutputParser()


class State(TypedDict):
    question: str
    name: Optional[str]
    link: Optional[str]
    bio: Optional[str]
    rag_docs: Optional[List]
    linkedin_profile: Optional[str]
    answer: Optional[str]
    linkedin_description: Optional[str]

def get_rag_context(state: State) -> State:
    state['rag_docs'] = retriever.invoke(state['question'])
    return state

def sufficient_checker(state: State) -> str:
    list = ['tell me about ', 'who is', 'give information about', 'linkedin']
    if any(word in state['question'].lower() for word in list):
        return 'use linkedin'
    else:
        return 'get_context'

def get_iit_bhubaneswar_profiles(state: State) -> list:
    matching_profiles = []
    names = ['Aryush Tripathi', 'Sumit Chaudhary', 'Anubhav Mishra', 'Poonam Patil']
    for name in names:
        if name in state['question']:
            search = GoogleSearch({
                "q": f"{name} linkedin IIT Bhubaneswar",
                "engine": "google",
                "api_key": os.getenv("SERPAPI_API_KEY")
            })
            results = search.get_dict()
            if "organic_results" in results:
                for result in results["organic_results"]:
                    profile = {
                        "name": result.get("title", "No Title"),
                        "bio": result.get("snippet", "No Bio Found"),
                        "link": result.get("link", "No Link")
                    }
                    matching_profiles.append(profile)
                    break
    return matching_profiles

def linkedin_scraper(state: State) -> State:
    response = get_iit_bhubaneswar_profiles(state)
    state['name'] = response[0]['name']
    state['bio'] = response[0]['bio']
    state['link'] = response[0]['link']
    state['linkedin_profile'] = response[0]['link']

    generation_prompt = PromptTemplate(
        template='''You are a helpful assistant. Read the following {bio} and give a short, clear description of the person without commentary.''',
        input_variables=['bio']
    )
    chain = generation_prompt | llm | output_parser
    text = chain.invoke({'bio': state['bio']})
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', '', text)
    text = re.sub(r'\+','',text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    state['linkedin_description'] = text
    return state

def generate_group_answers(state: State) -> State:
    context = "\n".join(doc.page_content for doc in (state['rag_docs'] or []))
    prompt = PromptTemplate(
        template='''You are a helpful assistant. Carefully read the following context and answer the question in a friendly way.Try to take out the answer from the context.Don't give opinion of yours.If you do not know the answer tell that you don't know the answer.
{context}
Question: {question}
Answer:''',
        input_variables=['context', 'question']
    )
    final_chain = RunnableParallel({'context': RunnablePassthrough(), 'question': RunnablePassthrough()}) | prompt | llm | output_parser
    reply = final_chain.invoke({'context': context, 'question': state['question']})
    state['answer'] = reply.strip().split('\n')[-1]
    return state

builder = StateGraph(State)
builder.add_node('context_receiver', get_rag_context)
builder.add_node('linkedin_scraper', linkedin_scraper)
builder.add_node('generate_answer', generate_group_answers)
builder.add_conditional_edges(START, sufficient_checker, {
    'get_context': 'context_receiver',
    'use linkedin': 'linkedin_scraper'
})
builder.add_edge('context_receiver', 'generate_answer')
builder.add_edge('generate_answer',END)
builder.add_edge('linkedin_scraper',END)
graph = builder.compile()


st.set_page_config(page_title="The गिरोह GPT", layout="centered")

st.markdown("""
    <style>
    /* Set the main background gradient */
    .stApp {
        background: linear-gradient(to right, #000000, #001F5F, #0040FF);
        background-attachment: fixed;
    }

    
    .search-bar input {
        width: 90%;
        padding: 15px;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        background-color: #D2AAAA; 
        color: black;
    }

    /* Button styling */
    .search-button {
        background-color: orange;
        color: black;
        border: none;
        padding: 8px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 15px;
        cursor: pointer;
        display: block;
        margin: 20px auto;
    }

    /* Title styling */
    .main-title {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }
    .title-the { color: orange; }
    .title-giroh { color: white; font-family: 'Devanagari'; }
    .title-gpt { color: orange; }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="main-title">
    <span class="title-the">THE</span> 
    <span class="title-giroh"> गिरोह </span> 
    <span class="title-gpt">GPT</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .description-text {
        font-size: 18px;
        text-align: center;
        color: #dcdcdc;
        margin-top: -10px;
        margin-bottom: 30px;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>

    <div class="description-text">
        This Chatbot is designed to give information about the whatsapp group "The गिरोह" and its members.
    </div>
""", unsafe_allow_html=True)

question = st.text_input(" ", placeholder="Ask me anything...", label_visibility="collapsed")

if st.button("Search"):
    if question:
        result = graph.invoke({'question': question})
        if "linkedin_description" in result:
            st.success(result["linkedin_description"])
        elif "answer" in result:
            st.success(result["answer"])
        else:
            st.info("No relevant answer found.")
    else:
        st.warning("Please enter a question.")
