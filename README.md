#  Stock Market Text Retrieval and Q&A using LangChain + Falcon-7B

---

##  Company Banners

<div align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="60" style="margin-right:40px;"/>
  <img src="https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain-stack.svg" alt="LangChain" height="60" style="margin-right:40px;"/>
  <img src="https://avatars.githubusercontent.com/u/98392786?s=200&v=4" alt="Chroma" height="60"/>
</div>

---

##  Overview

This notebook demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **Falcon-7B**, and **ChromaDB** to analyze stock market reports and provide context-aware answers.

You will:

* Download and load stock market text data
* Split documents into vectorized chunks
* Use Falcon-7B for natural language inference
* Query the database for relevant financial insights

---

##  Setup and Installation

Install the required libraries for embeddings, LLMs, and vector storage.

```python
!pip install sentence_transformers
!pip install -q langchain tiktoken chromadb pypdf transformers
!pip install -q accelerate bitsandbytes sentencepiece Xformers
!pip install langchain-community
```

---

##  Load and Prepare Dataset

Download the sample stock market data, unzip, and load into LangChain.

```python
!wget https://github.com/Shafi2016/Youtube/raw/main/stock_market_june_2023.zip -O stock_market_june_2023.zip
!unzip stock_market_june_2023.zip
```

Use **DirectoryLoader** to read text files and prepare the corpus.

```python
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader('./stock_market_june_2023/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()
```

---

##  Text Chunking

Split documents into overlapping text chunks for embedding and retrieval.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
```

---

##  Load Falcon-7B Model

Initialize the **Falcon-7B-Instruct** model pipeline for text generation.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, transformers

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
```

---

##  Create Vector Database

Use **HuggingFace embeddings** and **ChromaDB** to store document vectors.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
persist_directory = 'db'

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=persist_directory
)
vectordb.persist()
```

---

##  Initialize Retriever and QA Chain

Build a **retriever** and a **RetrievalQA** chain using LangChain.

```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipeline)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

---

##  Ask Questions

Define a helper function to process responses and display results.

```python
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
```

Ask the model a question about the dataset:

```python
query = "Could you please enumerate the companies that have been highlighted for their potential stock growth"
llm_response = qa_chain.invoke({"query": query})
process_llm_response(llm_response)
```

---

##  Output Example

```
Apple Inc.  
Tesla Motors  
Amazon Inc.  

Sources:  
./stock_market_june_2023/market_analysis_1.txt  
./stock_market_june_2023/market_trends.txt
```

---

##  Summary

This notebook shows a **complete RAG workflow**:

1. Load and prepare textual data
2. Embed using **HuggingFace**
3. Store vectors with **ChromaDB**
4. Retrieve and answer with **Falcon-7B**

---

**Author:** [Chiejina Chike Obinna](https://github.com/obinnachike)
**Frameworks:** Hugging Face • LangChain • ChromaDB • Falcon-7B
