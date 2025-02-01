import torch
from datasets import load_dataset
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from transformers import AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

data = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train")

data.to_csv("stanford_dataset.csv")

loader = CSVLoader(file_path='stanford_dataset.csv', encoding = 'UTF-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
model_name=modelPath, 
model_kwargs=model_kwargs, 
encode_kwargs=encode_kwargs 
)

db = FAISS.from_documents(docs, embeddings)

from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", padding=True, truncation=True, max_length=512)

pipe = pipeline(
     "text-generation", 
     model=model, 
     tokenizer=tokenizer,
     return_tensors='pt',
     max_length=512,
     model_kwargs={"torch_dtype": torch.bfloat16},
     device="cuda"
)

llm = HuggingFacePipeline(
     pipeline=pipe,
     model_kwargs={"temperature": 0.7, "max_length": 512},
)

qa = RetrievalQA.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=db.as_retriever()
)

qa.invoke("Write an educational story for young children.")