import io
from fastapi import FastAPI, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import PyPDF2
import uvicorn

app = FastAPI()

# Initialize embeddings and vectorstore
base_url = "http://ai-lab.sagitec.com:11434"
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest", base_url=base_url)
CHROMA_PATH = "./chroma_db"

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

llm = OllamaLLM(model="deepseek-r1:8b", base_url=base_url)

def process_and_store_pdf(contents, file_name):
    """Extracts text from a PDF, splits it into chunks, and stores them in ChromaDB."""
   # check for the exsisting docs in the db
    # existing_docs = vectorstore.get(where={"source": file_name})
    # if len(existing_docs["documents"]) > 0:
    #     return {"message": "File already processed and stored in ChromaDB", "chunks_stored": len(existing_docs["documents"])}
    # Extract text from PDF
    pdf_stream = io.BytesIO(contents)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Store in ChromaDB
    chunk_data = []
    for index, chunk_text in enumerate(chunks):
        metadata = {"source": file_name, "chunk_index": index, "category": "PDF"}
        chunk_data.append({"id": f"{file_name}_chunk_{index}", "text": chunk_text, "metadata": metadata})

    texts = [chunk["text"] for chunk in chunk_data]
    metadata = [chunk["metadata"] for chunk in chunk_data]
    ids = [chunk["id"] for chunk in chunk_data]

    vectorstore.add_texts(texts=texts, metadatas=metadata, ids=ids)
    return {"message": "File processed and stored successfully", "chunks_stored": len(chunks)}

@app.post("/upload/")
async def upload_file(file: UploadFile):
    contents = await file.read()
    response = process_and_store_pdf(contents, file.filename)
    return response

@app.get("/query/")
async def query_document(query: str):
    """Retrieves relevant chunks from ChromaDB and generates a response using the LLM."""
    # query_embedding = embeddings.embed_query(query)
    # retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)
    retrieved_docs = vectorstore.similarity_search_with_score(query=query, k=3)
    context = "\n".join([doc[0].page_content for doc in retrieved_docs])
    scores = [doc[1] for doc in retrieved_docs]

    prompt = f"""
    You are an AI assistant. Use the following context to answer the question:

    Context:
    {context}

    Question: {query}

    Answer:
    """
    response = llm.invoke(prompt)
    # print(scores)
    return {"query": query, "answer": response, "retrieved_context": context, "scores": scores}

@app.get("/view_all/")
async def view_all_documents():
    """Fetch all documents stored in ChromaDB."""
    all_docs = vectorstore.get()
    if not all_docs:
        return {"message": "No documents found in ChromaDB."}
    else:
        return {
            "documents": all_docs.get("documents", []),
            "metadatas": all_docs.get("metadatas", []),
            "ids": all_docs.get("ids", [])
        }
@app.delete("/delete_all/")
async def delete_all_documents():
    """Deletes all documents stored in ChromaDB."""
    try:
        vectorstore.delete_collection()
        return {"message": "All documents have been deleted from ChromaDB."}
    except Exception as e:
        return {"error": str(e)}

    
uvicorn.run(app, host="127.0.0.1", port=8200)
