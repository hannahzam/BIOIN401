import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import getpass
import os

from langchain_groq import ChatGroq

from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions

#from transformers import GPT2Tokenizer, TrainingArguments, Trainer
#from datasets import load_dataset

from TTS.api import TTS

import importlib
import requests
import time
import shutil
import gradio as gr


if "GROQ_API_KEY" not in os.environ:
     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
#if "PLAY_HT_USER_ID" not in os.environ:
#     os.environ["PLAY_HT_USER_ID"] = getpass.getpass("Enter your PlayHT User ID: ")
#if "PLAY_HT_API_KEY" not in os.environ:
#     os.environ["PLAY_HT_API_KEY"] = getpass.getpass("Enter your PlayHT API Key: ")
if "TAVUS_API_KEY" not in os.environ:
     os.environ["TAVUS_API_KEY"] = getpass.getpass("Enter your Tavus API Key: ")
if "SYNC_API_KEY" not in os.environ:
     os.environ["SYNC_API_KEY"] = getpass.getpass("Enter your Sync API Key: ")

# TTS from PlayHT
def speak(talk):
     load_dotenv()

     client = Client(
          user_id=os.getenv("PLAY_HT_USER_ID"),
          api_key=os.getenv("PLAY_HT_API_KEY"),
     )
     options = TTSOptions(voice="s3://voice-cloning-zero-shot/a1feef9e-6753-4d9a-b16a-f2814af26a87/original/manifest.json")
     # Open a file to save the audio
     with open("output3.wav", "wb") as audio_file:
          for chunk in client.tts(talk, options,  voice_engine='Play3.0-mini', protocol='http'):
          # Write the audio chunk to the file
               audio_file.write(chunk)

     client.close()

     '''
     device = "cuda" if torch.cuda.is_available() else "cpu"
     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
     # generate speech by cloning a voice using default settings
     tts.tts_to_file(text=talk,
                     file_path="output3.wav",
                     speaker_wav=r"Audio_250205024815.wav",
                     language="en")
     '''

# Generate video from Tavus API
def generate_avatar_video(text):
    tavus_api = os.getenv("TAVUS_API_KEY")
    try:
        url = "https://tavusapi.com/v2/videos"
        payload = {"replica_id": "rb17cf590e15", "script": text}
        headers = {"x-api-key": tavus_api, "Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        if response_data.get("status") != "queued":
            print("Failed to queue video generation.")
            return None, None
        video_id = response_data.get("video_id")
        video_check_url = f"https://tavusapi.com/v2/videos/{video_id}"
        return video_id, video_check_url
    except Exception as e:
        print(f"Error generating avatar video: {e}")
        return None, None
    
# generate lip-sync video from Sync.so
def generate_lip_sync(answer):
     sync_api = os.getenv("SYNC_API_KEY")
     try:
          url = "https://api.sync.so/v2/generate"

          payload = {
               "model": "lipsync-1.9.0-beta",
               "input": [
                    {
                         "type": "video",
                         "url": "https://synchlabs-public.s3.us-west-2.amazonaws.com/david_demo_shortvid-03a10044-7741-4cfc-816a-5bccd392d1ee.mp4"
                    },
                    {
                         "type": "audio",
                         "url": "https://synchlabs-public.s3.us-west-2.amazonaws.com/david_demo_shortaud-27623a4f-edab-4c6a-8383-871b18961a4a.wav"
                    }
               ],
               "options": {
                    "pads": [0, 5, 0, 0],
                    "speedup": 1,
                    "output_format": "mp4",
                    "sync_mode": "bounce",
                    "fps": 25,
                    "output_resolution": [1280, 720],
                    "active_speaker": True
               },
               
          }
          headers = {
               "x-api-key": sync_api,
               "Content-Type": "application/json"
          }

          response = requests.request("POST", url, json=payload, headers=headers)
          response_data = response.json()

          # status handling
          if response_data.get("status") == "FAILED": 
               print("Video generation failed.")
          elif response_data.get("status") == "REJECTED":
               print("Video generation rejected.")
          elif response_data.get("status") == "CANCELED":
               print("Video generation cancelled.")
          elif response_data.get("status") == "REJECTED":
               print("Video generation rejected.")

          video_url = response_data.get("outputUrl")
          print(video_url)

          while not video_url:   
               # wait for video
               time.sleep(5)
          
          save_video(video_url)

     except Exception as e:
          print(f"Error generating avatar video: {e}")


     return video_url


def save_video(url):
     response = requests.get(url, stream=True)
     response.raise_for_status() # was download successful?
     with open("answer.mp4", 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
     del response

    
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
     '''
     #while True: 
     print("Ask a question? or press enter to end conversation")
     question = input()
          #if not question:
          #     print("Hope to talk to you again sometime.")
          #     break
     '''
     relevant_docs = retriever.invoke(question)
     context = "\nExtracted documents:\n"
     context += "".join([f"Document {str(i)}:::\n" + str(doc) for i, doc in enumerate(relevant_docs)])
     prompt = prompt_template.invoke({"context": context, "question": question})
     chain = (
     llm
     | StrOutputParser()
     )
     answer = chain.invoke(prompt)
     # print(': ', answer, '\n')

     # have idle video running 
     yield answer, gr.update(value="https://drive.google.com/file/d/1let6bXm9EvBJUbHtnMahTvk-KLtSZRMf/view?usp=sharing", autoplay = True)

     # Generate avatar video
     video_id, video_check_url = generate_avatar_video(answer)
     if not video_id:
        return answer, gr.update(value=None, label="Error generating video.")

    # Initial response with loading message
     loading_message = gr.update(value="Video is being generated, please wait...", visible=True)
     yield answer, loading_message

     tavus_api = os.getenv("TAVUS_API_KEY")
    # Poll for video status every 5 seconds
     while True:
        time.sleep(5)
        headers = {"x-api-key": tavus_api}
        status_response = requests.get(video_check_url, headers=headers).json()
        print(status_response)

        if status_response.get("status") == "ready":
            download_url = status_response.get("download_url")
            mp4_download_url = download_url + ".mp4"  # Ensure .mp4 extension

            # Update the video component with the download URL
            yield answer, gr.update(value=mp4_download_url, visible=True)
            return  # Exit once the video is ready and displayed

        print("Video generation in progress...")

# Define Gradio interface
with gr.Interface(
    fn=main_conversation,
    inputs=[
        gr.Textbox(label="Your Input", placeholder="Type your message here..."),
    ],
    outputs=[
        gr.Textbox(label="Chatbot Response"),
        gr.Video(label="Avatar Video", autoplay=True)
    ],
    title="System Context Conversation Bot with Speaking Avatar",
    description="Interact with the chatbot, and receive a lifelike video response."
) as demo:
     demo.launch(share=True, debug=True)