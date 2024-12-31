# RAG-Based Chatbot Demo

This repository contains a demonstration of a **Retrieval-Augmented Generation (RAG)** pipeline. The chatbot combines document retrieval with a generative language model to provide contextually relevant and factually accurate responses. It includes two implementations:
1. A **basic RAG pipeline** for handling short queries and knowledge-based interactions.
2. An **advanced hierarchical retrieval and summarization pipeline** for handling long documents that exceed token limits.

---

## Features

- **Document Retrieval**: Uses FAISS and Sentence-BERT for fast and accurate document matching.
- **Generative Response**: Leverages BART to create human-like responses based on retrieved context.
- **Customizable Knowledge Base**: Easily adaptable to various domains, including banking, finance, and more.
- **Hierarchical Retrieval**: Processes long documents by chunking and summarizing before generation, enabling scalability beyond token limits.

---

## Setup

### Prerequisites

- Python 3.8 or above
- Required Python libraries:
  - `torch`
  - `transformers`
  - `sentence-transformers`
  - `faiss`
  - `numpy`
