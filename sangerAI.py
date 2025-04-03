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
import random

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
with ui.column().classes('items-center justify-center text-center w-full mt-12'):
    ui.label('🧠 SangerAI 🧠').classes('text-4xl font-bold')
    ui.label('Have a conversation with SangerAI, the virtual AI avatar created to be in the likeness of the famous Frederick Sanger! (this is a non-commercial, academic project)').classes('text-lg max-w-2xl text-gray-300')
ui.separator()

# dark mode 
ui.dark_mode().value = True

# Taking the user prompt after clicking the button and inputting it into LLM
async def ask():
     response_gallery = []
     no_message.set_text('')
     ask_button.disable()
     spinner_overlay.visible = True
     card_content.visible = False
     video.set_source('action/thinking.mp4')
     waiting_audio = ui.audio('thinking.mp3', autoplay=True, loop=True).classes('hidden')
     user_input = question.value
     # display relevant images to the user's query based on keyword 
     for fname in os.listdir('images'):
        if (fname[:-5].lower() in user_input.lower()):
            response_gallery.append('images/'+ fname)
     await asyncio.sleep(0.1)
     print(response_gallery)
     set_carousel_images(response_gallery)
     print('[DEBUG] Carousel rendered:', carousel)
     response = await run.cpu_bound(main_conversation, user_input)
     audio_answer = await speak(response)
     waiting_audio.delete()
     spinner_overlay.visible = False
     card_content.visible = True
     response_title.set_text("My answer...")
     response_label.set_text(response)
     response_audio = ui.audio(audio_answer, autoplay=True).classes('hidden')
     video.set_source('for lip sync/pseudo_ls.mp4')
     response_audio.on('ended', lambda _: (reset(), cleanup_audio(audio_answer, response_audio)))
     add_to_chat_history(user_input, response)

def reset():
     video.set_source('action/action_2.mp4')
     intro_images = ['images/intro.jpg', 'images/nobel.jpg', 'images/sequencing.jpg']
     set_carousel_images(intro_images)
     response_title.set_text("What would you like to know?")
     response_label.set_text('')
     ask_button.enable()

def set_carousel_images(images: list):
    """Clear and populate carousel slides with new image list."""
    if not images:
        print("No images to display.")
        return
    global carousel
    carousel.clear()
    random.shuffle(images)
    for src in images:
        with carousel:
            with ui.carousel_slide().classes('w-full flex justify-center items-center p-4 h-auto'):
                ui.image(src).classes('w-full h-auto')

def add_to_chat_history(user_msg, bot_msg):
    with chat_history_column:
        ui.label(f' Q: {user_msg}').classes('text-white font-semibold')
        ui.label(f' A: {bot_msg}').classes('text-blue-300')

def cleanup_audio(file_path, response_audio):
    response_audio.delete()
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Could not delete audio file: {e}")

intro_images = ['images/intro.jpg', 'images/nobel.jpg', 'images/sequencing.jpg']
carousel = None 

with ui.element().classes('flex flex-row gap-4 w-full'): 
     # Left: Image gallery and responses
     with ui.row().classes('w-full justify-center items-start gap-40'):
          with ui.column().classes('w-[700px]'):
               # --- Carousel ---
               with ui.carousel(animated=True, arrows=True, navigation=True).classes('w-full h-auto').props('autoplay=10000') as carousel:
                    set_carousel_images(intro_images)

               # --- Card Below Carousel ---
               with ui.card().classes('w-full mt-4 shadow-md'):
                    with ui.element().classes('absolute inset-0 bg-gray bg-opacity-80 flex justify-center items-center z-50') as spinner_overlay:
                         ui.spinner()
                    spinner_overlay.visible = False
                    with ui.row().classes('items-center justify-between p-4') as card_content:
                         with ui.column():
                              response_title = ui.label("Start a conversation by asking a question...").classes('text-xl font-semibold')
                              response_label = ui.label('')
               # 🧠 Expansion below the card
               with ui.expansion('Chat History', icon='history', value=False).classes('w-full max-w-full mt-2 border border-gray-500 rounded-lg'):
                    with ui.column().classes('p-2 gap-2 max-h-[300px] overflow-y-auto') as chat_history_column:
                         no_message = ui.label('No messages yet.').classes('text-gray-400').style('font-style: italic')

     # Right: Video avatar
          with ui.column().classes('w-[400px]'):
               video = ui.video('action/hello.mp4', autoplay=True, loop=True).classes('max-h-[700px] rounded shadow')

intro_audio = ui.audio('intro_audio.wav', autoplay=True).classes('hidden')
intro_audio.on('ended', lambda _: video.set_source('action/action_1.mp4'))

# input box for user question
with ui.element().classes(
        'fixed bottom-6 left-1/2 transform -translate-x-1/2 '
        'bg-gray-900 shadow-xl rounded-xl px-4 py-3 w-full max-w-screen-md z-50'
    ):
        with ui.row().classes('w-full items-center gap-2'):
            # user inputs question here
            question = ui.input(placeholder='Ask me a question...').props('outlined dense').props('clearable').classes('flex-1').props('outlined dense dark') \
     .classes('bg-gray-800 text-white rounded-lg')
            ask_button = ui.button("Ask", on_click=ask)

# bottom padding to scroll past the input
ui.element().classes('h-32')
        

# Run the NiceGUI app
ui.run(port=8080)