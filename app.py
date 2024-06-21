import PyPDF2
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

from filetype import guess
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image

# from unstructured.partition.pdf import partition_pdf

# from llama_parse import LlamaParse
# from dotenv import load_dotenv
# load_dotenv() 
# import os

# import re
from pdfminer.high_level import extract_pages, extract_text

# import streamlit

def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_text += extract_file_content(pdf.path)
        '''
        pdf_reader = PyPDF2.PdfReader(pdf.path)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
            '''
    return pdf_text

def convert_pdf_to_image(pdf_file): 
    return convert_from_path(pdf_file)

def convert_image_to_text(pdf_file):
    return image_to_string(pdf_file)

'''def get_pdf_elements(pdf_file):
    raw_pdf_elements = partition_pdf(
        filename=pdf_file,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=pdf_file,
    )
    return raw_pdf_elements'''

def detect_document_type(document_path):
    
    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf" 
    elif(guess_file.extension.lower() in image_types):
        file_type = "image"
    else:
        file_type = "unkown"
        
    return file_type

def extract_file_content(file_path):
    documents_content = ""
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        pdf_reader = PyPDF2.PdfReader(file_path)
        for page in pdf_reader.pages:
            documents_content += page.extract_text()
        '''raw_pdf_elements = get_pdf_elements(file_path)
        tables = []
        # texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                documents_content += (str(element))'''
        # scanned pdf where OCR required to extract text
        if (documents_content == ""):
            images = convert_pdf_to_image(file_path)
            for pg, img in enumerate(images):
                documents_content += convert_image_to_text(img)
    elif(file_type == "image"):
        documents_content +=convert_image_to_text(file_path)
    return documents_content

@cl.on_chat_start
async def on_chat_start():
    files = None #Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,# Optionally limit the file size
            timeout=600, # Set a timeout for user response,
            max_files=10,
        ).send()
   
    msg = cl.Message(content=f"Processing documents...")
    await msg.send()
    pdf_text = get_pdf_text(files)
        
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        Ollama(model="llama3"),
        # ChatOllama(model="gemma:7b"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing "
    for pdf in files:
        msg.content += f"`{pdf.name}` "
    msg.content += f" done. You can now ask questions!"
    await msg.update()
    #store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    #return results
    await cl.Message(content=answer, elements=text_elements).send()