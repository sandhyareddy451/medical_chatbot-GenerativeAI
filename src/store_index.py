from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, text_chunks, download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY



extracted_data = load_pdf_file(data = 'data')
text_splits = text_chunks (extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key = PINECONE_API_KEY)
index_name =  "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    
    documents= text_splits,
    index_name = index_name,
    embedding = embeddings
    
    
)