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


if "GROQ_API_KEY" not in os.environ:
     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
if "PLAY_HT_USER_ID" not in os.environ:
     os.environ["PLAY_HT_USER_ID"] = getpass.getpass("Enter your PlayHT User ID: ")
if "PLAY_HT_API_KEY" not in os.environ:
     os.environ["PLAY_HT_API_KEY"] = getpass.getpass("Enter your PlayHT API Key: ")


def speak(talk):
     load_dotenv()

     client = Client(
          user_id=os.getenv("PLAY_HT_USER_ID"),
          api_key=os.getenv("PLAY_HT_API_KEY"),
     )
     options = TTSOptions(voice="s3://voice-cloning-zero-shot/a1feef9e-6753-4d9a-b16a-f2814af26a87/original/manifest.json")
     # Open a file to save the audio
     with open("output.wav", "wb") as audio_file:
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

#def tokenize_function(examples):
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer(examples["Answer"], padding="max_length", truncation=True)

def load_llm():
     '''
     # trying to train the llm
     t_dataset = load_dataset("csv", data_files="training_set.csv")
     e_dataset = load_dataset("csv", data_files="eval_set.csv")
     training_dataset = t_dataset.map(tokenize_function, batched=True)
     evalu_dataset = e_dataset.map(tokenize_function, batched=True)

     training_args = TrainingArguments(
          output_dir="test_trainer", 
          per_device_eval_batch_size=1,
          per_device_train_batch_size=1,
          gradient_accumulation_steps=4
     )
     '''

     llm = ChatGroq(
          model="llama-3.3-70b-versatile",
          temperature=0,
          max_tokens=None,
          timeout=None,
          max_retries=2
     )

     #llm.train(training_dataset, **training_args) 

     #trainer = Trainer(
     #     model=llm,
     #     args=training_args,
     #     train_dataset=training_dataset,
     #     eval_dataset=evalu_dataset
     #)
     #trainer.train()

     """
     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", low_cpu_mem_usage = True)
     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", padding=True, truncation=True, max_length=512, low_cpu_mem_usage = True)
     

     pipe = pipeline(
          task="text-generation", 
          model=model, 
          tokenizer=tokenizer,
          pad_token_id=tokenizer.eos_token_id, 
          do_sample=True,
          return_full_text=False,
     )

     llm = HuggingFacePipeline(
          pipeline=pipe,
          model_kwargs={"temperature": 0.1, "max_length": 512, "max_new_tokens": 500},
     )
     """
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

def main():
     db = create_vector_store(data_dir='Fred Sanger Data collection')
     llm = load_llm()
     prompt_template = create_prompt_template()
     retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
     #while True: 
     print("Ask a question? or press enter to end conversation")
     question = input()
          #if not question:
          #     print("Hope to talk to you again sometime.")
          #     break
     relevant_docs = retriever.invoke(question)
     context = "\nExtracted documents:\n"
     context += "".join([f"Document {str(i)}:::\n" + str(doc) for i, doc in enumerate(relevant_docs)])
     prompt = prompt_template.invoke({"context": context, "question": question})
     chain = (
     llm
     | StrOutputParser()
     )
     answer = chain.invoke(prompt)
     print(': ', answer, '\n')
     speak(answer)

main()