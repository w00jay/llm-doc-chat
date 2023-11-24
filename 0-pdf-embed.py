import os
# import torch
# from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# def generate_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).numpy()  # Convert to numpy array

# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# model = AutoModel.from_pretrained("bert-large-uncased")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1200,
    chunk_overlap  = 10,
    length_function = len,
    is_separator_regex = False,
)

# Initialize the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")  # all-MiniLM-L6-v2

# Directory containing PDF files
pdf_directory = "../input"  # "."

# Process each PDF file
doc_info_mapping = {}
for i, pdf_file in enumerate(os.listdir(pdf_directory)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        
        docs = text_splitter.create_documents(
            texts = [text],
            metadatas = [{"file_path": pdf_path}],
        )
        print(docs[0])

        # Initialize or create a new Chroma DB
        chroma = Chroma.from_documents(
            documents = docs,
            persist_directory="./test/chroma-all-mpnet-base-v2",
            embedding = embedding_function,
        )
        
        # Save the mapping
        doc_info_mapping[i] = os.path.join(pdf_directory, pdf_file)
        chroma.persist()


# Save the mapping to a file
import json
with open("doc_info_mapping.json", "w") as file:
    json.dump(doc_info_mapping, file)
