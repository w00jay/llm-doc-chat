import os
import json
import logging
import concurrent.futures
import re

# import torch
# from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


# def extract_metadata_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     metadata = reader.metadata
#     cleaned_metadata = {key.lstrip('/'): value for key, value in metadata.items()}
#     return cleaned_metadata


def remove_surrogates(text):
    # Regex to match surrogate pairs
    surrogate_regex = re.compile(r"[\uD800-\uDFFF]")
    return surrogate_regex.sub("", text)


def extract_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Remove surrogate pairs
            page_text = remove_surrogates(page_text)

            # Normalize text encoding to handle special characters
            page_text = page_text.encode("utf-8", "replace").decode("utf-8")
            text += page_text + "\n"

    # Convert metadata to a regular dictionary and ensure all values are simple data types
    metadata = {}
    metadata["file_path"] = pdf_path if pdf_path else None
    if reader.metadata is not None:
        for key, value in reader.metadata.items():
            key = key.lstrip("/")
            # Convert value to string to ensure compatibility with Chroma
            metadata[key] = str(value) if value is not None else None

    return text, metadata


def process_pdf(pdf_file):
    all_docs = []
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        log.info(f"Processing {pdf_path}...")

        text, metadata = extract_from_pdf(pdf_path)
        # metadata = extract_metadata_from_pdf(pdf_path)

        docs = text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        for doc in docs:
            all_docs.append(doc)

        if all_docs:
            print(all_docs[0].page_content)
            log.info(f"Finished with {pdf_path}.")
            return all_docs
        else:
            log.info(f"No documents found in {pdf_path}.")
            return None


# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1200,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)


# Initialize the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-mpnet-base-v2"
)  # all-MiniLM-L6-v2


# Directory containing PDF files
pdf_directory = "../input"  # "../input-test"  # "."

# Define the number of workers
num_workers = 7

# Process PDFs in parallel
all_docs = []
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_pdf = {
        executor.submit(process_pdf, pdf_file): pdf_file
        for pdf_file in os.listdir(pdf_directory)
    }

    for future in concurrent.futures.as_completed(future_to_pdf):
        docs = future.result()

        if docs:
            all_docs.extend(docs)


log.info(f"Finished processing all PDFs. Got {len(all_docs)} documents.")


# Initialize or create a new Chroma DB with all documents
chroma = Chroma.from_documents(
    documents=all_docs,
    persist_directory="./test/chroma-all-mpnet-base-v2",
    embedding=embedding_function,
)
chroma.persist()


# # Save the mapping to a file
# doc_info_mapping = {i: doc.metadata['file_path'] for i, doc in enumerate(all_docs)}
# with open("doc_info_mapping.json", "w") as file:
#     json.dump(doc_info_mapping, file)
