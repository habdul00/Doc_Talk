from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app = app = Flask(__name__, template_folder='templates')


# Load OpenAI API Key from file
with open('file.txt', 'r') as f:
    openai_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

# Define functions for setting up vector database and processing prompt
def setup_vectordb():
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    loader = DirectoryLoader('./journals/', glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    return persist_directory, embedding

def process_prompt(prompt, persist_directory, embedding):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    llm_response = qa_chain(prompt)
    return llm_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/response', methods=['POST'])
def get_response():
    prompt = request.json['prompt']
    persist_directory, embedding = setup_vectordb()
    llm_response = process_prompt(prompt, persist_directory, embedding)
    response_text = llm_response['result']
    return jsonify({'result': response_text})

if __name__ == '__main__':
    app.run(debug=True)
