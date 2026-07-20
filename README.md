<div align="center">

  <img width="501" height="112" alt="IntelliDocs Logo" src="https://github.com/user-attachments/assets/245a3c4d-4707-48e3-931d-396c109c7555" />

  A local RAG (Retrieval-Augmented Generation) pipeline for querying your documents.


  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
  ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
  ![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=databricks&logoColor=FF3621)

</div>

---

## Features

* **Modular Architecture**: Clean separation between retrieval, embedding, and response generation modules.
* **Vector Store Management**: Script to rebuild and sync vector stores (`rebuild_vector_store.py`).
* **Evaluation Framework**: Built-in evaluation metrics to track retrieval performance.
* **Interactive Chat**: Query your docs seamlessly using `chat.py`.

## Project Structure

```text
├── data/                    # Raw document storage
├── docs/                    # Documentation
├── eval/                    # Evaluation metrics and logs
├── notebook/                # Jupyter notebooks for experimentation
├── rag/                     # Core RAG implementation
├── chat.py                  # CLI chat interface
├── rebuild_vector_store.py  # Vector store indexer
└── requirements.txt         # Dependencies
