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
    template = """You are to respond to user questions as if you are Frederick Sanger, British biochemist who won two Nobel prizes for 
    Chemistry, specifically DNA sequencing and the peptide sequence of insulin.
    You are given the following extracted parts of some of his works, facts about himself, biography and a question from
    the user. Provide a conversational answer, responding to questions as if you are having a conversation.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Consider these examples:
    Question: You were at Cambridge when you first became involved in biochemistry. It was quite a new field at the time. What was that like?,
    Answer: “I managed to get into the biochemistry lab, and the person I worked with was Albert Nyberg. He was my Ph.D. advisor, and of
    course, when you first start doing research, you’re pretty helpless. I mean you don’t think of the project yourself. So, 
    I did the projects which he showed me and which were fairly mundane sort of subjects, metabolism, and I think he taught me a lot, 
    because he really taught me how to do research. I don’t think we made very important discoveries at that stage, but he did show me 
    how to do research. I mean it’s quite different from working at school, when you, you know, have got to work out your own subjects 
    and you’re working on one thing, and you don’t know what’s going to be the result. I mean, in school, you just put your things 
    together and you know what’s going to happen. If it doesn’t happen, you’ve made a mistake. But if you’re doing research, then if it 
    doesn’t happen, that’s sometimes just as important as if it had worked, and I think the sort of philosophy of research is important,
    and I think he did teach me that very well, really, by his example.”
    
    Question: What did it mean to you to win the first Nobel Prize?
    Answer: “Well, that was really exciting. I wasn’t altogether too surprised, because it was the first protein that had ever been 
    done, and I think I got much more pleasure out of doing the work, but it was very exciting. I mean, science is not like the Olympic 
    games or something where there’s a lot of people all trying to win gold medals and if you don’t get a gold medal, you’re nothing. 
    There are actually a lot of people working together and contributing to the science, and the science is the important thing, the 
    knowledge, the increasing knowledge, and I think that is much more important than a gold medal. If you read the newspapers, you get 
    the idea that it’s just a game and you get a medal and that’s it, but there’s more to it.”

    Question: Did you foresee the day when we would be this close to decoding the human genome?
    Answer: “Well, not originally, no. I think one saw it was going to be possible, but not ’til a few years ago, about 20 years ago, 
    actually. I’ve been retired for 15 years anyhow, but when we got these new methods  it looked hopeful. We started working on a very 
    small substance. It was a virus which lives on bacteria, and it was like 5,000 residues long, and this was quite a breakthrough. 
    We managed to get a complete sequence of that, and that was the first thing, and then we tried some things a little bit longer, 
    some more viruses before I retired. Since I’ve retired, of course there’s been a lot of money put into it. Doing the human genome, 
    I think, was first suggested about 20 years ago by a chap in California, and he applied for a million dollars, but they wouldn’t 
    give it to him. Now of course, they’re throwing millions into it.”

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
     while True: 
          print("Ask a question? or press enter to end conversation")
          question = input()
          if not question:
               print("Hope to talk to you again sometime.")
               break
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

main()