"""RAG Knowledge System demo

This demo shows how to build a Retrieval-Augmented Generation (RAG) pipeline:
1. **Ingest** — load documents from a source, embed them, store in pgvector
2. **Query** — retrieve relevant chunks and pass them as context to an LLM

Architecture:
    Source (Wikipedia) → Ingestor → Embeddings → PGVector
                                                      ↓
    Question → Retriever → Context → Prompt → LLM → Answer

Usage:
    python src/rag_demo.py

Environment variables:
    DATABASE_URL  PostgreSQL connection string with pgvector enabled
                  e.g. postgresql+psycopg://user:pass@localhost:5432/mydb
"""

import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

# Add src directory to path so relative ingestor imports work
sys.path.insert(0, str(Path(__file__).parent))

from ingestors import WikipediaIngestor

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

temperature = 0.1
collection_name = "rag_documents"

# ---------------------------------------------------------------------------
# Embeddings (local, no API key required)
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Vector store (PostgreSQL + pgvector)
# ---------------------------------------------------------------------------

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise EnvironmentError(
        "DATABASE_URL environment variable is not set.\n"
        "Example: postgresql+psycopg://user:pass@localhost:5432/mydb"
    )

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=database_url,
    use_jsonb=True,
)

# ---------------------------------------------------------------------------
# Ingestor registry
# Students: add a new entry here to make your ingestor appear in the UI.
# ---------------------------------------------------------------------------

INGESTORS = {
    "Wikipedia": WikipediaIngestor(),
    # "Files":     FileIngestor(directory="data/"),   # <-- Activity 5, Part 2
    # "URL":       URLIngestor(),                     # <-- Activity 5, Part 3
}

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

ollama_model = "qwen2.5:3b"
ollama_client = ChatOllama(model=ollama_model, temperature=temperature)

llamacpp_server = os.environ.get("PERDRIZET_URL", "localhost:8502")

if llamacpp_server.startswith("localhost") or llamacpp_server.startswith("127."):
    llamacpp_api_key = os.environ.get("LLAMA_API_KEY", "dummy")
    llamacpp_base_url = f"http://{llamacpp_server}/v1"
else:
    llamacpp_api_key = os.environ.get("PERDRIZET_API_KEY")
    llamacpp_base_url = f"https://{llamacpp_server}/v1"

llamacpp_client = ChatOpenAI(
    base_url=llamacpp_base_url,
    api_key=llamacpp_api_key,
    timeout=120.0,
    model="gpt-oss-20b",
    temperature=temperature,
)

llamacpp_model = "gpt-oss-20b"

# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using only the "
        "provided context. If the context does not contain enough information "
        "to answer, say so honestly.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_sources(docs) -> str:
    sources = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        preview = doc.page_content[:200].replace("\n", " ")
        sources.append(f"[{i}] {title}\n    {source}\n    \"{preview}...\"")
    return "\n\n".join(sources)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def ingest_documents(topic: str, ingestor_name: str) -> str:
    """Load documents from the selected source and store them in pgvector."""
    if not topic.strip():
        return "Please enter a topic."

    ingestor = INGESTORS[ingestor_name]

    try:
        docs = ingestor.load(topic.strip())
    except Exception as e:
        return f"Error loading documents: {e}"

    if not docs:
        return "No documents found for that topic."

    try:
        vector_store.add_documents(docs)
    except Exception as e:
        return f"Error storing documents: {e}"

    # Count unique source articles
    unique_sources = {doc.metadata.get("source", "") for doc in docs}
    return (
        f"Ingested {len(docs)} chunks from {len(unique_sources)} article(s).\n"
        f"Source: {ingestor.source_type} | Topic: {topic}"
    )


def clear_collection() -> str:
    """Delete all documents from the vector store collection."""
    global vector_store
    try:
        vector_store.delete_collection()
        # Re-initialise so the store is ready for new ingestions
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=database_url,
            use_jsonb=True,
        )
        return "Collection cleared. Ready for new ingestion."
    except Exception as e:
        return f"Error clearing collection: {e}"


def query_rag(question: str, backend: str, k: int) -> tuple[str, str]:
    """Retrieve relevant chunks and generate a grounded answer."""
    if not question.strip():
        return "Please enter a question.", ""

    llm = ollama_client if backend == "Ollama" else llamacpp_client
    retriever = vector_store.as_retriever(search_kwargs={"k": int(k)})

    # Fetch docs separately so we can display them as sources
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "No relevant documents found. Try ingesting some content first.",
            "",
        )

    chain = (
        {"context": lambda _: _format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke(question)
    except Exception as e:
        return f"Error generating answer: {e}", ""

    sources_text = _format_sources(retrieved_docs)
    return answer, sources_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG Knowledge System") as demo:

    gr.Markdown("""
    # RAG Knowledge System

    Build a searchable knowledge base from Wikipedia articles, then ask questions
    grounded in the ingested content.

    **How it works:**
    1. **Ingest** — search Wikipedia by topic, chunk the articles, embed and store them
    2. **Query** — your question retrieves the most relevant chunks, which the LLM uses to answer
    """)

    with gr.Tabs():

        # ------------------------------------------------------------------
        # Tab 1: Ingest
        # ------------------------------------------------------------------
        with gr.Tab("1. Ingest documents"):
            gr.Markdown("""
            Type a Wikipedia topic to fetch and store articles in the knowledge base.
            You can ingest multiple topics — they accumulate in the same collection.
            """)

            with gr.Row():
                with gr.Column():
                    topic_input = gr.Textbox(
                        label="Wikipedia topic",
                        placeholder="e.g. Python programming language",
                        value="Python programming language",
                    )
                    ingestor_radio = gr.Radio(
                        choices=list(INGESTORS.keys()),
                        value=list(INGESTORS.keys())[0],
                        label="Document source",
                        info="Students: add new sources in src/ingestors/ and register them in INGESTORS above.",
                    )
                    with gr.Row():
                        ingest_btn = gr.Button("Ingest", variant="primary")
                        clear_btn = gr.Button("Clear collection", variant="stop")

                with gr.Column():
                    ingest_status = gr.Textbox(label="Status", lines=4, interactive=False)

            ingest_btn.click(
                fn=ingest_documents,
                inputs=[topic_input, ingestor_radio],
                outputs=[ingest_status],
            )
            clear_btn.click(
                fn=clear_collection,
                inputs=[],
                outputs=[ingest_status],
            )

        # ------------------------------------------------------------------
        # Tab 2: Query
        # ------------------------------------------------------------------
        with gr.Tab("2. Query knowledge base"):
            gr.Markdown("""
            Ask a question. The system retrieves the most relevant chunks from the
            knowledge base and uses them as context for the answer.
            """)

            with gr.Row():
                backend_selector = gr.Radio(
                    choices=["Ollama", "llama.cpp"],
                    value="Ollama",
                    label="Model backend",
                    info=f"Ollama: {ollama_model} | llama.cpp: {llamacpp_model} @ {llamacpp_base_url}",
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Chunks to retrieve (k)",
                    info="More chunks = more context, but slower and more expensive.",
                )

            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What is Python used for?",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")

                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")

            with gr.Accordion("Sources", open=False):
                sources_output = gr.Textbox(label="Retrieved chunks", lines=12, interactive=False)

            ask_btn.click(
                fn=query_rag,
                inputs=[question_input, backend_selector, k_slider],
                outputs=[answer_output, sources_output],
            )

    gr.Markdown("""
    ---

    ## Key concepts

    | Component | Role |
    |-----------|------|
    | `HuggingFaceEmbeddings` | Turns text into vectors (numbers) that capture meaning |
    | `PGVector` | Stores vectors in PostgreSQL; finds nearest neighbours fast |
    | `WikipediaIngestor` | Loads + chunks Wikipedia articles |
    | Retriever | Finds the *k* most similar chunks to the question |
    | RAG chain | Injects retrieved chunks as context into the LLM prompt |

    **Next step:** Try Activity 5 to add new document sources!
    """)


if __name__ == "__main__":
    demo.launch()
