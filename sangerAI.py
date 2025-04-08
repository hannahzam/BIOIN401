from nicegui import ui, run, app
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

import asyncio
import random

import time
import edge_tts


async def speak(talk):
     start = time.time()
     file_name = "answer.wav"
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

# dark mode 
ui.dark_mode().value = True

@ui.page('/')
def home():
     ui.dark_mode().value = True
     # Dark theme styles
     ui.add_head_html("""
<style>
  body {
    background-color: #0f0f0f;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    margin: 0;
  }

  /* Animation classes */
  .reveal {
    opacity: 0;
    transform: translateY(40px);
    transition: opacity 0.8s ease, transform 0.8s ease;
  }

  .reveal.active {
    opacity: 1;
    transform: translateY(0);
  }

  /* Styling */
  .hero-title {
    font-size: 6rem;
    font-weight: 600;
    color: #ffffff;
  }

  .hero-sub {
    font-size: 2rem;
    color: #aaaaaa;
    max-width: 48rem;
    line-height: 2.2rem;
    font-weight: 400;
  }

  .chat-button {
    background-color: #222;
    color: #f0f0f0;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    border: none;
    transition: background 0.2s ease;
  }

  .chat-button:hover {
    background-color: #333;
  }

  .banner-img {
    width: 100%;
    object-fit: contain;
    opacity: 0.2;
  }

     html {
    scroll-behavior: smooth;
  }
  body {
    background-color: #0f0f0f;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    margin: 0;
  }
  .nav-link {
    text-decoration: none;
    color: white;
    font-size: 1.1rem;
    font-weight: 500;
  }
</style>

<script>
  const revealOnScroll = () => {
    const reveals = document.querySelectorAll('.reveal');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
        }
      });
    }, { threshold: 0.2 });

    reveals.forEach(el => observer.observe(el));
  };

  window.addEventListener('DOMContentLoaded', revealOnScroll);
</script>
""")

# Sticky navbar
     with ui.row().classes('fixed top-0 left-0 w-full bg-black px-8 py-6 items-center justify-between z-50 shadow-sm'):
          ui.label('🧬').classes('text-3xl font-bold text-white')  # larger logo text

          with ui.row().classes('gap-10'):
               ui.link('About', '#about').classes('text-white text-lg no-underline nav-link')
               ui.link('References', '#references').classes('text-white text-lg no-underline nav-link')

# Hero section with scroll animations
     with ui.column().classes('w-full h-screen items-center justify-center text-center').style('padding-top: 4rem;'):
          ui.label('SangerAI').classes('hero-title reveal')
          ui.label(
               'Have a conversation with SangerAI, a virtual chat avatar\n'
               'designed in the likeness of Fred Sanger.'
          ).classes('hero-sub reveal')
          ui.button('Press here to start chatting').classes('mt-8 chat-button reveal').on('click', lambda: ui.navigate.to('/chat'))

# Banner image with scroll-in animation
     ui.image('banner.jpg') \
     .classes('banner-img reveal mt-[-1rem]')

     with ui.column().classes('items-start px-10 py-20 w-full').props('id=about'):

          ui.label('About Dr. Frederick Sanger') \
               .classes('text-3xl font-bold text-white reveal')

          # Small margin between title and quote
          with ui.row().classes('items-baseline reveal'):
               ui.label(
                    '“It is like a voyage of discovery into unknown lands, seeking not for new territory '
                    'but for new knowledge.” '
               ).classes('italic text-lg text-gray-300')

               ui.label('- Fred Sanger').classes('text-lg font-bold text-white')

          # Body paragraph fades in last
          ui.label(
               'Dr. Fred Sanger was my scientist of choice for the “Dead Scientist Avatar” project. '
               'He’s a two-time Nobel prize winner in Chemistry for two incredibly important discoveries: '
               'the amino acid sequence of insulin and the sequencing of DNA.'
          ).classes('mt-6 text-base text-gray-300 leading-relaxed reveal')

     with ui.column().classes('py-32 items-center').props('id=references'):
          ui.label('📚 References Section').classes('text-3xl font-bold')

# /auth page
@ui.page('/auth')
def auth_page():
    ui.add_head_html("""
<style>
  body {
    background-color: #0f0f0f;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
  }
  .input {
    padding: 0.75rem;
    border-radius: 8px;
    border: 1px solid #444;
    background-color: #1a1a1a;
    color: white;
    width: 300px;
    font-size: 1rem;
  }
  .submit-btn {
    background-color: #333;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    border: none;
    margin-left: 1rem;
    cursor: pointer;
  }
  .submit-btn:hover {
    background-color: #444;
  }
  a {
    color: #9ac6ff;
    text-decoration: underline;
  }
</style>
""")
    if ui.session.client.storage.get('groq_api_key'):
        ui.open('/chat')
        return

    with ui.column().classes('w-full h-screen items-center justify-center text-center gap-4'):
        ui.label('Enter your Groq API Key').classes('text-2xl font-bold')
        ui.html('Visit <a href="https://console.groq.com/keys" target="_blank">GroqCloud</a> to get your own API key') \
            .classes('text-md text-gray-400')

        api_input = ui.input(placeholder='Enter your API key...').classes('input')

        def handle_submit():
            key = api_input.value.strip()
            if key:
                # ✅ Store it in the session (not os.environ)
                ui.session.client.storage['groq_api_key'] = key
                ui.open('/chat')

        ui.button('Submit', on_click=handle_submit).classes('submit-btn')

@ui.page('/chat')
def chat():
     ui.dark_mode().value = True
     if not ui.session.client.storage.get('groq_api_key'):
        ui.open('/auth')
        return
     # Taking the user prompt after clicking the button and inputting it into LLM
     async def ask():
          ui.run_javascript("document.getElementById('waitingAudio').pause()")
          response_gallery = []
          no_message.set_text('')
          ask_button.disable()
          spinner_overlay.visible = True
          card_content.visible = False
          video.set_source('action/thinking.mp4')
          thinking_audio = ui.audio('thinking.mp3', autoplay=True, loop=True).classes('hidden')
          user_input = question.value
          response = await run.cpu_bound(main_conversation, user_input)
          audio_answer = await speak(response)
          # display relevant images to the user's query based on keyword 
          for fname in os.listdir('images'):
               if (fname[:-5].lower() in response.lower()):
                    response_gallery.append('images/'+ fname)
          set_carousel_images(response_gallery)
          thinking_audio.delete()
          spinner_overlay.visible = False
          card_content.visible = True
          response_title.set_text("My answer...")
          response_label.set_text(response)
          response_audio = ui.audio(audio_answer, autoplay=True).classes('hidden')
          video.set_source('idle/pseudo_ls.mp4')
          response_audio.on('ended', lambda _: (reset(), cleanup_audio(audio_answer, response_audio)))
          add_to_chat_history(user_input, response)
          print(emotion_detection(response))

     def reset():
          video.set_source('action/action_2.mp4')
          set_carousel_images(intro_images)
          response_title.set_text("What would you like to know?")
          response_label.set_text('')
          ui.run_javascript("document.getElementById('waitingAudio').play()")
          ask_button.enable()

     def set_carousel_images(images: list):
          """Clear and populate carousel slides with new image list."""
          if not images:
               print("No images to display.")
               return
          carousel.clear()
          carousel.value = len(images)
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

     def emotion_detection(question):
          llm = load_llm()
          template = """According to these emotional categories: happy, sad, angry, neutral; give a single
          word answer corresponding to the category this question would be under as an emotional response
          to the question. Just respond with a single word, no added symbols. If you are unsure, just respond with
          neutral.

          Question: {question}
          Answer in Markdown:"""
          prompt_template = PromptTemplate(template=template, input_variables=["question"])
          prompt = prompt_template.invoke({"question": question})

          chain = (
          llm
          | StrOutputParser()
          )
          answer = chain.invoke(prompt)
    
          return str(answer)
     
     with ui.row().classes('top-0 left-0 w-full bg-gray px-8 py-6 items-center justify-between z-50 shadow-sm'):
        ui.link('←  Back to Home', '/').classes('text-white text-lg font-medium no-underline hover:text-gray-400')
     
     with ui.element().classes('flex flex-row gap-4 w-full'): 
          # Left: Image gallery and responses
          with ui.row().classes('w-full justify-center items-start gap-40'):
               with ui.column().classes('w-[700px]'):
                    # --- Carousel ---
                    intro_images = ['images/intro1.jpg', 'images/intro2.jpg']
                    with ui.carousel(animated=True, arrows=True, navigation=True).classes('w-full h-auto').props('autoplay=10000') as carousel:
                         for src in intro_images:
                              with ui.carousel_slide().classes('w-full flex justify-center items-center p-4 h-auto'):
                                   ui.image(src).classes('w-full h-auto')

                    # --- Card Below Carousel ---
                    with ui.card().classes('w-full h-70 mt-4 shadow-md'):
                         with ui.element().classes('absolute inset-0 bg-gray bg-opacity-80 flex justify-center items-center z-50') as spinner_overlay:
                              ui.spinner().props('size=50px color=white')
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
                    video = ui.video('action/action_1.mp4', autoplay=True, loop=True).classes('max-h-[700px] rounded shadow')

          waiting_audio = ui.audio('waiting.mp3', autoplay=True, loop=True).props('id=waitingAudio').classes('hidden')

     # input box for user question
     with ui.element().classes(
          'fixed bottom-6 left-1/2 transform -translate-x-1/2 '
          'bg-gray-800 shadow-xl rounded-xl px-4 py-3 w-full max-w-screen-md z-50'
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