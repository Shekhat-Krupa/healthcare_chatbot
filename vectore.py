# import os
# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from PIL import Image
# import pytesseract

# DATA_PATH = 'data/'
# DB_FAISS_PATH = 'vectorstore/db_faiss'


# # Function to perform OCR on image-based PDFs (if necessary)
# def extract_text_with_ocr(pdf_path):
#     # In case of image-based PDFs, use OCR to extract text from each page
#     text = pytesseract.image_to_string(Image.open(pdf_path))
#     return text


# # Create vector database from PDFs
# def create_vector_db():
#     # Check if the data directory contains any PDF files
#     pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
#     print(f"PDF files found: {pdf_files}")
    
#     if not pdf_files:
#         print("No PDF files found in the data directory.")
#         return

#     # Load PDFs from the directory using PyPDFLoader
#     loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
#     documents = loader.load()

#     if not documents:
#         print("No documents loaded from the PDFs.")
#         return

#     print(f"Loaded {len(documents)} documents.")

#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Extract the page content for encoding
#     texts_content = [doc.page_content for doc in texts]

#     # Debugging: Print the number of text chunks and sample text
#     print(f"Number of text chunks: {len(texts_content)}")
#     if texts_content:
#         print(f"Sample text chunk: {texts_content[0]}")
#     else:
#         print("No valid text content found. Attempting OCR if necessary.")

#         # Optionally apply OCR if no text was extracted
#         ocr_texts = []
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(DATA_PATH, pdf_file)
#             ocr_text = extract_text_with_ocr(pdf_path)
#             if ocr_text.strip():
#                 ocr_texts.append(ocr_text)

#         # If OCR produced results, use them
#         if ocr_texts:
#             texts_content = ocr_texts
#             print(f"Extracted text using OCR from {len(ocr_texts)} PDFs.")
#         else:
#             print("OCR failed to extract text as well. Exiting.")
#             return

#     # Filter out empty or invalid texts
#     texts_content = [text for text in texts_content if text.strip()]

#     if not texts_content:
#         print("No valid text content to encode after filtering.")
#         return

#     # Load sentence-transformers model for embedding generation
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#     # Generate embeddings for the text chunks
#     embeddings = model.encode(texts_content, show_progress_bar=True)

#     # Create FAISS vector store
#     db = FAISS.from_texts(texts_content, embeddings)
#     db.save_local(DB_FAISS_PATH)
#     print("Vector database successfully created and saved locally.")


# if __name__ == "__main__":
#     create_vector_db()


# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# DATA_PATH = 'data/'
# DB_FAISS_PATH = 'vectorstore/db_faiss'

# # Create vector database
# def create_vector_db():
#     # Load PDFs from the directory
#     loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
#     documents = loader.load()

#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Extract the texts for encoding
#     texts_content = [doc.page_content for doc in texts]
    
#     # Debugging: Print some information about the loaded texts
#     print(f"Number of text chunks: {len(texts_content)}")
#     print(f"Sample text chunk: {texts_content[0] if texts_content else 'No text'}")

#     # Filter out empty texts
#     texts_content = [text for text in texts_content if text.strip()]

#     if not texts_content:
#         print("No valid text content to encode.")
#         return  # Exit if there's no valid text

#     # Load sentence-transformers model for embedding generation
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#     # Generate embeddings
#     embeddings = model.encode(texts_content, show_progress_bar=True)  # Encode texts into embeddings

#     # Create the FAISS vector store
#     db = FAISS.from_texts(texts_content, embeddings)
#     db.save_local(DB_FAISS_PATH)

# if __name__ == "__main__":
#     create_vector_db()


from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    # Load PDFs from the directory
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load sentence-transformers model for embedding generation
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Extract the texts for encoding
    texts_content = [doc.page_content for doc in texts]  # Extract the content of each split document

    # Generate embeddings
    embeddings = model.encode(texts_content, show_progress_bar=True)  # Encode texts into embeddings

    # Create the FAISS vector store
    db = FAISS.from_texts(index=texts_content, embedding_function=embeddings)  # Pass texts and their embeddings
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()


# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter 

# DATA_PATH = 'data/'
# DB_FAISS_PATH = 'vectorstore/db_faiss'

# # Create vector database
# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH,
#                              glob='*.pdf',
#                              loader_cls=PyPDFLoader)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)

# if __name__ == "__main__":
#     create_vector_db()
