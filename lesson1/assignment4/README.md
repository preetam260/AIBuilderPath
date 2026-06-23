In order to build a rag based chatbot from scratch, we need 4 things.
- A place to store documents
- Convert query text into embeddings 
- Find relevant chunks 
- LLM that answers based on the retrieved context

Working:
- files are obtained from the folder, and chunked using character-based 
chunking with a fixed size of 300 characters.
- each chunk is converted into vector embeddings using sentence transformer.
- embeddings are stored inside faiss vector index.
- flat indexing is used: perform exact similarity search which is accurate 
but slower but sufficient for the content length of the docs.
- if it is unable to retrieve relevent context it returns I dont know.

Limitations: 
- since character based chunking was used related information mightve split 
across different chunks
- if the document consists of ApartmentServiceRule and instead of that if my 
query is phrased differently retrieval quality may be affected.

