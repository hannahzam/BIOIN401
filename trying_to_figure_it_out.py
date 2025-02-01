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

import getpass
import os

from langchain_groq import ChatGroq

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

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
          model="mixtral-8x7b-32768",
          temperature=0,
          max_tokens=None,
          timeout=None,
          max_retries=2
     )
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
    prompt_template = PromptTemplate.from_template("""Use the provided {context} to make
    a response to the user's {question}. If you don't know the answer, respond with "I do not know".
    Answer: 
    """)
    
    return prompt_template


def main():
     db = create_vector_store(data_dir='data')
     llm = load_llm()
     prompt_template = create_prompt_template()
     retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
     print("Chatbot for PDF files initialized, ready to query...")
     question = input("> ")
     relevant_docs = retriever.invoke(question)
     context = "\nExtracted documents:\n"
     context += "".join([f"Document {str(i)}:::\n" + str(doc) for i, doc in enumerate(relevant_docs)])
     prompt = prompt_template.invoke({"context": context, "question": question})
     answer = llm.invoke(prompt)
     print(': ', answer, '\n')


main()