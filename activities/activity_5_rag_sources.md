# Activity 5: Extending the RAG knowledge system

**Objective:** Add new document sources to the RAG system by implementing the `BaseIngestor` interface.

**Duration:** 45-60 minutes

---

## Overview

The RAG demo uses a modular ingestor system. Each ingestor knows how to:
1. **Load** documents from some source (Wikipedia, local files, a URL, …)
2. **Split** them into chunks suitable for embedding

All ingestors share the same interface so the rest of the system doesn't need to care where the documents came from.

```
Source                  Ingestor             Vector store
------                  --------             ------------
Wikipedia topic    →    WikipediaIngestor → PGVector (PostgreSQL)
Local files        →    FileIngestor      ↗
URL                →    URLIngestor       ↗
```

By the end of this activity you will have implemented a `FileIngestor` that lets users ingest local `.txt` and `.md` files — and it will automatically appear as an option in the demo UI.

---

## Setup

### 1. Start the RAG demo

Make sure your LLM backend is running (Ollama or llama.cpp), then:

```bash
python src/rag_demo.py
```

Open the Gradio interface and confirm the **Ingest** tab shows "Wikipedia" as the only source.

### 2. Create your activity file

Create `activities/my_ingestors.py` for testing outside the demo:

```python
"""Activity 5: Testing new ingestors."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestors import WikipediaIngestor
```

---

## Part 1: Explore the existing system

### Step 1: Read the base interface

Open `src/ingestors/base.py`. Note:
- `source_type` — a property returning the display name shown in the UI radio button
- `load(source)` — the only method you need to implement; returns a list of `Document` objects

### Step 2: Read the Wikipedia ingestor

Open `src/ingestors/wikipedia.py`. The full implementation is only ~25 lines:
- It wraps `WikipediaLoader` to fetch articles by search query
- It passes the raw documents through `RecursiveCharacterTextSplitter` so they become small chunks

### Step 3: Test it in your activity file

Add this to `activities/my_ingestors.py` and run it:

```python
ingestor = WikipediaIngestor(load_max_docs=1)  # just 1 article to keep it quick
docs = ingestor.load("Large language model")

print(f"Loaded {len(docs)} chunks")
print(f"\nFirst chunk:\n{docs[0].page_content[:300]}")
print(f"\nMetadata: {docs[0].metadata}")
```

**Questions to consider:**
- How large is each chunk in characters?
- What metadata fields does `WikipediaLoader` attach?
- What happens if you search for a topic that does not exist on Wikipedia?

---

## Part 2: Implement FileIngestor

**Goal:** Create an ingestor that loads `.txt` and `.md` files from a local directory.

### Step 1: Create the ingestor file

Create `src/ingestors/file.py`:

```python
"""File ingestor — loads local .txt and .md files."""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class FileIngestor(BaseIngestor):
    """Ingestor that loads text files from a local directory."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # TODO: Store chunk_size and chunk_overlap
        # TODO: Create a RecursiveCharacterTextSplitter with those parameters
        pass

    @property
    def source_type(self) -> str:
        # TODO: Return a human-readable name (e.g. "Files")
        pass

    def load(self, source: str) -> list[Document]:
        """Load all .txt and .md files from *source* directory.

        Args:
            source: Path to a directory containing text files.

        Returns:
            List of split Document chunks.
        """
        # TODO: Use DirectoryLoader to load files matching "**/*.txt" and "**/*.md"
        # Hint: DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
        # Hint: Load both patterns and combine the results before splitting
        pass
```

### Step 2: Export it from the package

Open `src/ingestors/__init__.py` and add your new class:

```python
from .base import BaseIngestor
from .wikipedia import WikipediaIngestor
from .file import FileIngestor          # add this line

__all__ = ["BaseIngestor", "WikipediaIngestor", "FileIngestor"]
```

### Step 3: Create some test files

```bash
mkdir -p data/docs
echo "LangChain is a framework for building applications powered by large language models." > data/docs/langchain.txt
echo "pgvector is a PostgreSQL extension that adds vector similarity search capabilities." > data/docs/pgvector.txt
```

### Step 4: Test your ingestor

Add to `activities/my_ingestors.py`:

```python
from ingestors.file import FileIngestor

file_ingestor = FileIngestor()
docs = file_ingestor.load("data/docs")

print(f"Loaded {len(docs)} chunks from local files")
for doc in docs:
    print(f"  - {doc.metadata.get('source', 'unknown')}: {doc.page_content[:80]}")
```

### Success criteria

- [ ] `FileIngestor` extends `BaseIngestor`
- [ ] `source_type` returns a meaningful string
- [ ] `load()` returns a non-empty list of chunks for a valid directory
- [ ] Both `.txt` and `.md` files are picked up

### Hints

<details>
<summary>Click to reveal hints</summary>

1. `DirectoryLoader` signature: `DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True})`
2. To load two patterns, create two loaders and concatenate the raw docs before splitting
3. `loader.load()` returns a list of `Document` objects — combine them with `+`
4. You can ignore files that don't exist rather than raising an error by passing `silent_errors=True` to `DirectoryLoader`

</details>

---

## Part 3: Register the ingestor in the demo

### Step 1: Open `src/rag_demo.py`

Find the `INGESTORS` dictionary near the top of the file:

```python
INGESTORS = {
    "Wikipedia": WikipediaIngestor(),
    # "Files":     FileIngestor(directory="data/"),   # <-- Activity 5, Part 2
    # "URL":       URLIngestor(),                     # <-- Activity 5, Part 3
}
```

### Step 2: Add your FileIngestor

```python
from ingestors.file import FileIngestor   # add to imports at top

INGESTORS = {
    "Wikipedia": WikipediaIngestor(),
    "Files": FileIngestor(),
}
```

### Step 3: Verify in the UI

Restart the demo (`python src/rag_demo.py`). You should see a new **"Files"** option in the ingestor radio on the Ingest tab. Select it, enter a directory path (e.g. `data/docs`), and click Ingest.

Then switch to the Query tab and ask a question about the content of your files.

---

## Part 4 (stretch): Implement URLIngestor

**Goal:** Fetch and ingest a web page by URL.

### Skeleton

Create `src/ingestors/url.py`:

```python
"""URL ingestor — loads a web page from a given URL."""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class URLIngestor(BaseIngestor):
    """Ingestor that loads content from a URL."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # TODO: implement
        pass

    @property
    def source_type(self) -> str:
        # TODO: implement
        pass

    def load(self, source: str) -> list[Document]:
        """Fetch and chunk the page at *source* URL.

        Args:
            source: A full URL, e.g. "https://docs.python.org/3/tutorial/"
        """
        # TODO: Use WebBaseLoader(web_paths=[source]) to fetch the page
        # TODO: Split the loaded documents and return the chunks
        pass
```

### Tips
- LangChain's `WebBaseLoader` uses `BeautifulSoup` under the hood — install it with `pip install beautifulsoup4`
- The `source` argument in the Ingest tab will be a full URL (students paste any URL they like)
- The metadata will include the page URL, which is useful to display as a citation in query results

---

## Key takeaways

1. **Interface > implementation** — the Gradio UI only knows about `BaseIngestor`; adding a new source is just one new file + one dict entry
2. **Chunking strategy matters** — chunk size affects retrieval quality; too large and you retrieve irrelevant noise, too small and you lose context
3. **Metadata is your citation system** — `Document.metadata` carries source information that surfaces in the "Sources" panel after querying
4. **The same embeddings model processes all sources** — as long as text is in English (or the model's language), the source type is irrelevant to the retriever

**Next step:** Try ingesting multiple sources on the same topic (e.g. Wikipedia + a local notes file) and see how the answers change!
