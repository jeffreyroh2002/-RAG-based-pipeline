import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Models
retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
generator_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')

# 2. Suppose we have a list of knowledge base documents
knowledge_docs = [
    "Document A: As of December 31, 2024, the interest rate for savings accounts is 2.5% per annum for balances below $10,000.",
    "Document B: Savings accounts with balances over $10,000 earn an interest rate of 3.0% per annum.",
    "Document C: Our premium savings account offers 4.0% annual interest for VIP members.",
]

# 3. Embed and Index Documents with FAISS
doc_embeddings = retrieval_model.encode(knowledge_docs)
dim = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(doc_embeddings)

def retrieve_docs(query, top_k=2):
    """Retrieve top_k relevant documents for the given query."""
    query_embedding = retrieval_model.encode([query])
    scores, indexes = faiss_index.search(query_embedding, top_k)
    return [knowledge_docs[i] for i in indexes[0]]

# 4. RAG-like Inference
def rag_inference(user_query):
    # Retrieve top relevant documents
    relevant_texts = retrieve_docs(user_query, top_k=2)
    
    # Combine the retrieved text for context
    context_text = " ".join(relevant_texts)
    prompt = f"Context: {context_text}\n\nUser Query: {user_query}\nAnswer:"
    
    # Generate the response using a seq2seq model (Bart, T5, etc.)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 5. Test the pipeline
if __name__ == "__main__":
    user_query = "What are the interest rates for savings accounts as of today?"
    response = rag_inference(user_query)
    print("User Query:", user_query)
    print("RAG-based Response:", response)