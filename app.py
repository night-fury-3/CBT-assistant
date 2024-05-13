from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_wtf.csrf import CSRFProtect

from dotenv import load_dotenv
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

app = Flask(__name__)

csrf = CSRFProtect(app)

text_splitter = CharacterTextSplitter(separator = "\n\n", chunk_size=2000, chunk_overlap=200, length_function = len)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

folder_list = ['PMI']

pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index(PINECONE_INDEX)

@app.route('/')
def home():
    return render_template('index.html', folder_list=folder_list)
            
@app.route('/uploadDocuments', methods=['POST'])
@csrf.exempt
def uploadDocuments():
    uploaded_files = request.files.getlist('files[]')
    folder = request.form['folder']

    print("Length of files =>", len(uploaded_files))
    print("Folder =>", folder)
    
    if len(uploaded_files) > 0:    
        try:
            for file in uploaded_files:
                file.save(f"uploads/{file.filename}")

                loader = PyPDFLoader(f"uploads/{file.filename}")
                    
                data = loader.load()

                texts = text_splitter.split_documents(data)
                metadata = {"source": folder}

                for text in texts:
                    vector = embeddings.embed_query(text.page_content)

                    id = str(uuid.uuid4())

                    pc_index.upsert(vectors=[{"id": id, "values": vector, "metadata": metadata}])

                os.remove(f"uploads/{file.filename}")

            return {'success': "ok"}
        except:
            return {"success": "bad"}
    else:
        return {"success": "bad"}


@app.route('/getFolder', methods=['POST'])
@csrf.exempt
def getFolder():
    query = request.form['query']

    query_vector = embeddings.embed_query(query)

    result = pc_index.query(vector=query_vector, top_k=1, include_metadata=True)

    folder = result['matches'][0]['metadata']['source']

    print("Folder ID =>", folder)

    return folder


if __name__ == '__main__':
    app.run(debug=True, port=5006)