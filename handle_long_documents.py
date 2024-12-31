import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load Models
retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summarization_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
summarization_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

# 2. Sample Long Document and Query
long_document = """
    Document Part 1: Financial report for Q2 2023 shows revenue of $20M. Net profit is $5M.
    Document Part 2: Additional details include operating expenses of $10M and taxes of $5M.
    Document Part 3: Future outlook projects a 10% growth in revenue for Q3 2023.
    """
query = "What is the net profit for Q2 2023?"

# 3. Preprocess the Long Document into Chunks
document_chunks = [
    "Financial report for Q2 2023 shows revenue of $20M. Net profit is $5M.",
    "Additional details include operating expenses of $10M and taxes of $5M.",
    "Future outlook projects a 10% growth in revenue for Q3 2023.",
]

# 4. Embed Document Chunks and Query
chunk_embeddings = retrieval_model.encode(document_chunks)
query_embedding = retrieval_model.encode([query])

# 5. Use FAISS to Retrieve Relevant Chunks
dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(chunk_embeddings)

_, indexes = faiss_index.search(query_embedding, k=2)  # Retrieve top-2 relevant chunks
retrieved_chunks = [document_chunks[i] for i in indexes[0]]

# 6. Summarize Retrieved Chunks
def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

summaries = [summarize_text(chunk) for chunk in retrieved_chunks]

# 7. Generate Final Answer Based on Summarized Context
final_context = " ".join(summaries)
generation_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
generation_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')

prompt = f"Context: {final_context}\n\nQuery: {query}\nAnswer:"
inputs = generation_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
outputs = generation_model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
answer = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 8. Output the Final Answer
print("Query:", query)
print("Final Answer:", answer)
