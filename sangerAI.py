from nicegui import ui, run
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

from TTS.api import TTS

import re
import asyncio

# get Groq API Key
if "GROQ_API_KEY" not in os.environ:
     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

def speak(talk):
     device = "cuda" if torch.cuda.is_available() else "cpu"
     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
     # generate speech by cloning a voice using default settings
     file_name = "response.wav"
     tts.tts_to_file(text=talk,
                     file_path=file_name,
                     speaker_wav=r"Audio_250205024815.wav",
                     language = "en")
     return file_name
     
def create_vector_store(data_dir):
    '''Create a vector store from PDF files'''
    # define what documents to load
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db
     
def load_llm():

     llm = ChatGroq(
          model="llama-3.3-70b-versatile",
          temperature=0,
          max_tokens=None,
          timeout=None,
          max_retries=2
     )
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
     db = create_vector_store(data_dir='Fred Sanger Data collection')
     llm = load_llm()
     prompt_template = create_prompt_template()
     retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
     relevant_docs = retriever.invoke(question)
     context = "\nExtracted documents:\n"
     context += "".join([f"Document {str(i)}:::\n" + str(doc) for i, doc in enumerate(relevant_docs)])
     prompt = prompt_template.invoke({"context": context, "question": question})
     chain = (
     llm
     | StrOutputParser()
     )
     answer = chain.invoke(prompt)
     return str(answer)


# Web App Interface via NiceGUI

# labelling the window SangerAI
ui.label('SangerAI').classes('text-3xl')

# Taking the user prompt after clicking the button and inputting it into LLM
async def ask():
     user_input = question.value
     # display relevant images to the user's query based on keyword 
     for fname in os.listdir('images'):
        if (fname[:-4] in user_input):
        # Display Image
            print(fname)
            ui.image('images/' + fname)
            break
     response = await run.cpu_bound(main_conversation, user_input)
     audio = await run.cpu_bound(speak, response)
     response_label.set_text(response)
     ui.audio(audio, autoplay=True)

# input box for user question
question = ui.input(label="Ask me a question.", placeholder= 'Type something...')
# hitting the button sends user query to the LLM
ui.button("Ask", on_click=ask).classes("col-span-full")

# display LLM response 
with ui.card().classes("col-span-full"):
     ui.markdown("My answer...")
     ui.separator()
     response_label = ui.label('')
        

# Run the NiceGUI app
ui.run(port=8080)