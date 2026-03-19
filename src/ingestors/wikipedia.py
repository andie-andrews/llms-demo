"""Wikipedia ingestor.

Fetches Wikipedia articles by topic and splits them into chunks
suitable for embedding and vector storage.
"""

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class WikipediaIngestor(BaseIngestor):
    """Ingestor that fetches Wikipedia articles by topic name."""

    def __init__(self, load_max_docs: int = 3, chunk_size: int = 500, chunk_overlap: int = 50):
        self.load_max_docs = load_max_docs
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def source_type(self) -> str:
        return "Wikipedia"

    def load(self, source: str) -> list[Document]:
        """Fetch Wikipedia articles matching *source* and return chunks.

        Args:
            source: Wikipedia search query / topic name,
                    e.g. "Python programming language".

        Returns:
            List of text chunks as LangChain Documents.
        """
        loader = WikipediaLoader(query=source, load_max_docs=self.load_max_docs)
        docs = loader.load()
        return self._splitter.split_documents(docs)
