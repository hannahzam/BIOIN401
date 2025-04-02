from nicegui import ui, run, app
from fastapi import Request
import os

import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import getpass
import os

from langchain_groq import ChatGroq

import re
import asyncio

import time
import edge_tts

# get Groq API Key
if "GROQ_API_KEY" not in os.environ:
     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

async def speak(talk):
     start = time.time()
     file_name = "response.wav"
     communicate = edge_tts.Communicate(talk, voice="en-GB-RyanNeural")
     await communicate.save(file_name)
     end = time.time()
     print("execution time for tts: ", str(end-start))
     return file_name

def cleanup_audio(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Could not delete audio file: {e}")
     
def create_vector_store(data_dir, index_path='vectorstore_index'):
    '''Create a vector store from PDF files'''
    start = time.time()
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(index_path)  # ⬅️ Save the index to disk
    end = time.time()
    print("Vector store created and saved.")
    print("execution time for vector database: ", str(end-start))
    return db

def load_or_create_vector_store(data_dir, index_path='vectorstore_index'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    if os.path.exists(index_path):
        print("Loading existing vector store...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Index not found. Creating vector store...")
        return create_vector_store(data_dir, index_path)
     
def load_llm():
     start = time.time()
     llm = ChatGroq(
          model="llama-3.3-70b-versatile",
          temperature=0,
          max_tokens=None,
          timeout=None,
          max_retries=2
     )
     end = time.time()
     print("execution time for llm loading: ", str(end-start))
     return llm

def create_prompt_template():
    # prepare the template we will use when prompting the AI
    template = """You are to respond to user questions as if you are Frederick Sanger, British biochemist who won two Nobel prizes for 
    Chemistry, specifically DNA sequencing and the peptide sequence of insulin.
    You are given a question from the user and using the relevant context, provide a conversational answer to the question.
    If you don't know the answer or the user does not provide a question, just say "Hmm, I'm not sure." Do not try to make up a question or an answer
    and do not repeat yourself within your answer. Do not include unnecessary symbols or a header to your answer. Just respond to the question.

    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    prompt_template = PromptTemplate(template=template, input_variables=["question", "context"])
    
    return prompt_template

def main_conversation(question):
     db = load_or_create_vector_store(data_dir='Fred Sanger Data collection')
     llm = load_llm()
     prompt_template = create_prompt_template()
     retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
     relevant_docs = retriever.invoke(question)
     context = "\nExtracted documents:\n"
     context += "".join([f"Document {str(i)}:::\n" + str(doc) for i, doc in enumerate(relevant_docs)])
     prompt = prompt_template.invoke({"context": context, "question": question})
     start = time.time()
     chain = (
     llm
     | StrOutputParser()
     )
     answer = chain.invoke(prompt)
     end = time.time()
     print("execution time for llm answer: ", str(end-start))
     return str(answer)



# Web App Interface via NiceGUI

# labelling the window SangerAI
ui.label('SangerAI').classes('text-3xl')

# Taking the user prompt after clicking the button and inputting it into LLM
async def ask():
     ask_button.disable()
     video.set_source('action/thinking.mp4')
     waiting_audio = ui.audio('thinking.mp3', autoplay=True, loop=True).classes('hidden')
     user_input = question.value
     # display relevant images to the user's query based on keyword 
     for fname in os.listdir('images'):
        if (fname[:-4] in user_input):
        # Display Image
            print(fname)
            image.set_source('images/' + fname)
            break
     response = await run.cpu_bound(main_conversation, user_input)
     audio = await speak(response)
     waiting_audio.delete()
     response_label.set_text(response)
     response_audio = ui.audio(audio, autoplay=True).classes('hidden')
     video.set_source('for lip sync/pseudo_ls.mp4')
     response_audio.on('ended', lambda _: (reset(), cleanup_audio(audio)))

def reset():
     video.set_source('action/action_2.mp4')
     image.set_source('images/intro.jpg')
     ask_button.enable()

# video of sanger with supplementary photo beside it 
with ui.row().classes('w-full items-center justify-center gap-4'):
     # plays the intro video, only should play once
     video = ui.video('action/hello.mp4', autoplay=True, loop=True).classes('w-96')
     intro_audio = ui.audio('intro_audio.wav', autoplay=True).classes('hidden')
     intro_audio.on('ended', lambda _: video.set_source('action/action_1.mp4'))
     image = ui.image('images/intro.jpg').style('width: 50%')

# input box for user question
with ui.row().classes('w-full items-center justify-center'):
     question = ui.input(label="Ask me a question.", placeholder= 'Type something...').props('clearable') \
        .props('outlined') \
        .classes('text-xl w-96')  # Large text and fixed width
     # hitting the button sends user query to the LLM
     ask_button = ui.button("Ask", on_click=ask)

# display LLM response 
with ui.card().classes("col-span-full"):
     ui.markdown("My answer...")
     ui.separator()
     response_label = ui.label('')
        

# Run the NiceGUI app
ui.run(port=8080)