# LLM-DOC-CHAT

This is a set of applications for supplementing LLM's chatbot
with a local document embeddings from your PDFs and EPUBs.


## Installation

I have no requirements.txt.  Sorry...  I'm a bad person.
Install a lot of dependencies...  But do this in a `virtualenv`
unless you want to be sorry later.

My favorite way is to use direnv and use `layout python` in this folder,
then `direnv allow` to provision your virtualenv that you don't have to
`deactivate` later.


## Usage

### Create your own embeddings from your own PDF and EPUB files

- Copy your source documents into a folder and specify it in `input_file_path` in `0-pdf-embed.py`.
- Run `python 0-pdf-embed.py`.  This will take some time.  In VScode, `SQLite3 Editor`` by yy0931 will
let you search it later w/in the editor.
- If the ingestion into Chroma DB fails, you will have to process the docs again.
- FAISS should work too.
- Feel free to adjust chunk size, as 1200 is too big for a lot of OSS LLMs.

### Test by querying the DB w/ LLM

- Run `python 1-chat.py` and ask questions.  It should return `relevant` document chunks.
- Feel free to set `k=` as needed.

### Use UI to chat

- Run `streamlit run 2-convo.py` for better experience.  W/ 1200-chunk size, OpenAI works but not the others.
- Please share your improvements.
