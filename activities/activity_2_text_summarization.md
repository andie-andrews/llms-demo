# Activity 2: Text summarization script

**Objective:** Build a Python script that uses an LLM to summarize text documents, applying prompting techniques from Lesson 46.

**Duration:** 45-60 minutes

---

## Overview

In this activity, you'll create a practical text summarization tool. You'll start with a basic implementation and progressively improve it by applying prompting techniques. This will help you understand how to build reliable LLM applications.

---

## Part 1: Basic implementation (15 minutes)

### Step 1: Create the script

Create a new file: `src/text_summarizer.py`

Start with this template:

```python
'''Text summarization using an LLM.

Usage:
    python src/text_summarizer.py input.txt
'''

import sys

# TODO: Import your LLM library
# Example: from langchain_ollama import ChatOllama
# Example: from openai import OpenAI

def summarize_text(text, model):
    '''Summarize the given text using an LLM.'''
    
    # TODO: Create a basic prompt
    prompt = f"Summarize this text: {text}"
    
    # TODO: Call the LLM
    # response = model.invoke(prompt)
    
    # TODO: Return the summary
    return response

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/text_summarizer.py input.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Read the input file
    with open(filename, 'r') as f:
        text = f.read()
    
    print(f"Input text: {len(text)} characters")
    print(f"Summarizing...\n")
    
    # TODO: Initialize your LLM model
    # model = ChatOllama(model='qwen2.5:3b')
    
    # Get summary
    summary = summarize_text(text, model)
    
    print("Summary:")
    print(summary)

if __name__ == '__main__':
    main()
```

### Step 2: Create test documents

Create a `test_texts/` directory with sample documents:

```bash
mkdir -p test_texts
```

**Sample 1:** Create `test_texts/short_article.txt`:
```
The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine. 
Fleming noticed that a mold growing on a bacterial culture plate had killed the 
surrounding bacteria. This accidental observation led to the development of the 
first widely used antibiotic. Penicillin saved countless lives during World War II 
and continues to be an essential medication today. Fleming shared the 1945 Nobel 
Prize in Physiology or Medicine for this groundbreaking discovery.
```

**Sample 2:** Create `test_texts/long_article.txt` (find a longer text, 500+ words)

### Step 3: Test your basic implementation

```bash
python src/text_summarizer.py test_texts/short_article.txt
```

**Observe:**
- Does it work?
- Is the summary good?
- What's missing?

---

## Part 2: Improve with prompting techniques (20 minutes)

Now enhance your script by applying techniques from Lesson 46.

### Technique 1: Be specific and add structure

**Update your prompt:**

```python
def summarize_text(text, model, max_words=100):
    prompt = f"""Summarize the following text.

Text:
{text}

Requirements:
- Maximum {max_words} words
- Focus on the main ideas
- Use clear, concise language

Summary:"""
    
    # Call LLM and return response
```

**Test again:** Does the output improve?

---

### Technique 2: Add context

If you're summarizing specific types of documents (e.g., news, research, technical docs), add context:

```python
def summarize_text(text, model, max_words=100, document_type='general'):
    contexts = {
        'news': 'This is a news article. Focus on who, what, when, where, why.',
        'research': 'This is a research document. Focus on objectives, methods, and findings.',
        'technical': 'This is technical documentation. Focus on key features and important details.',
        'general': 'This is a general text document.'
    }
    
    context = contexts.get(document_type, contexts['general'])
    
    prompt = f"""Summarize the following text.

Context: {context}

Text:
{text}

Requirements:
- Maximum {max_words} words
- Focus on the main ideas
- Use clear, concise language

Summary:"""
```

**Update main() to accept document type:**

```python
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/text_summarizer.py input.txt [type]")
        print("Types: news, research, technical, general")
        sys.exit(1)
    
    filename = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else 'general'
    
    # ... rest of the code
    summary = summarize_text(text, model, document_type=doc_type)
```

---

### Technique 3: Use constraints

Add specific constraints based on your needs:

```python
prompt = f"""Summarize the following text.

Text:
{text}

Requirements:
- Maximum {max_words} words
- Write in bullet points
- Focus on actionable information
- Do not include speculative statements
- Use present tense

Summary:"""
```

**Experiment:** Try different constraint combinations for different document types.

---

## Part 3: Advanced features (15 minutes)

### Option A: Multi-level summaries

Add support for different summary lengths:

```python
def summarize_text(text, model, summary_type='short'):
    lengths = {
        'brief': 50,
        'short': 100,
        'medium': 200,
        'detailed': 400
    }
    
    max_words = lengths.get(summary_type, 100)
    
    # ... rest of prompt
```

### Option B: Key points extraction

Create a separate function for extracting key points:

```python
def extract_key_points(text, model, num_points=5):
    prompt = f"""Extract the {num_points} most important points from this text.

Text:
{text}

Format your response as a numbered list.
Be concise - each point should be one sentence.

Key Points:"""
    
    # Call LLM and return
```

### Option C: Handle long documents

For documents longer than the model's context window, add chunking:

**Challenge:** How would you handle a document that's too long for the LLM's context window?

**Approach:**
1. Split the text into smaller chunks (e.g., 2000 characters each)
2. Summarize each chunk individually
3. Combine the chunk summaries
4. Create a final summary of the summaries

**Hints:**
- Use `text.split()` to split by words (better than splitting mid-sentence)
- Track the size of each chunk as you build it
- Remember to handle the last chunk after the loop
- You might want separate functions: `chunk_text()` and `summarize_long_text()`

**Pseudocode structure:**

```python
def chunk_text(text, max_chunk_size=2000):
    '''Split text into chunks of approximately max_chunk_size characters.'''
    # TODO: Split text.split() into word list
    # TODO: Build chunks by adding words until reaching max_chunk_size
    # TODO: Return list of chunk strings
    pass

def summarize_long_text(text, model, max_words=100):
    '''Summarize text that may be longer than context window.'''
    # TODO: If text is short enough, summarize directly
    # TODO: Otherwise, chunk the text
    # TODO: Summarize each chunk
    # TODO: Combine chunk summaries and create final summary
    pass
```

---

## Part 4: Testing and refinement (10 minutes)

### Test with different documents

1. **Short article** (100-200 words) - should be straightforward
2. **Medium article** (500-1000 words) - tests your prompts
3. **Long document** (2000+ words) - tests chunking (if implemented)
4. **Technical document** - tests context handling

---


## Key takeaways

- **Clear prompts with structure** produce more consistent results
- **Context matters** - the same text may need different summaries for different audiences
- **Constraints help** - specify what you want and what you don't want
- **Iteration is essential** - test with real documents and refine
- **Edge cases are important** - long documents, short documents, technical content
- **Prompt engineering is practical** - small changes in prompts can significantly affect output quality

---

## Next steps

- Apply these techniques to other text processing tasks (classification, extraction, translation)
- Explore advanced techniques from Lesson 47 (chain-of-thought, self-consistency, prompt chaining)
- Build a more complex application combining multiple LLM calls

---

## Resources

**Example implementations to reference:**
- `src/ollama_chatbot.py` - Shows LangChain usage
- `src/llamacpp_chatbot.py` - Shows OpenAI client usage

**Documentation:**
- [LangChain Chat Models](https://python.langchain.com/docs/integrations/chat/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Python Library](https://github.com/ollama/ollama-python)
