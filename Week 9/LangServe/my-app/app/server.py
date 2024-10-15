from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes


from langchain_community.llms import VLLMOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant, QdrantVectorStore
from langchain.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
from typing import Any
from pydantic import BaseModel
#######################
app = FastAPI()


# constants
vllm_docker_address = "http://localhost:8000/v1" 
vllm_model = "deepseek-ai/deepseek-llm-7b-chat"
embedding_model = 'mxbai-embed-large'
qdrant_docker_address = 'http://localhost:6333'
wikipedia_page_name = '2024 NBA playoffs'
collection_name = 'NBAplayoffs2024'


# prompt
rag_prompt_template = '''\
<｜begin▁of▁sentence｜>You are a helpful assistant. You answer User Query based on provided Context. If you can't answer the User Query with the provided Context, say you don't know.

User: 

Question: {question}

Context: {context}

Assistant: \
'''
rag_prompt = PromptTemplate.from_template(rag_prompt_template).with_config({'run_name':'rag_prompt'})

# llm
llm = VLLMOpenAI(
    openai_api_key="fake_api_key",  
    openai_api_base=vllm_docker_address,  
    model_name=vllm_model
).with_config({'run_name':'vllm_deepseek-llm-7b-chat'})

# embedding model
embeddings = OllamaEmbeddings(model=embedding_model)


# vector store & retriever
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
client = QdrantClient(qdrant_docker_address)
if collection_name not in list([x.name for x in client.get_collections().collections]):
    loader = WikipediaLoader(wikipedia_page_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=qdrant_docker_address, 
        collection_name=collection_name
    )
else:
    vectorstore = QdrantVectorStore(
        client = client,
        collection_name=collection_name,
        embedding=embeddings,
        )
retriever = vectorstore.as_retriever(search_kwargs  = {'k':1}).with_config({'run_name':'retriever_mxbai-embed-large'})


# RAG chain
rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context= lambda x : '\n\n'.join([ y.metadata['summary'] for y in x["context"] ])    )
        | rag_prompt | llm
    ).with_config({'run_name':'RAG_chain'})

class Input(BaseModel):
    question: str

class Output(BaseModel):
    output: str


#######################
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(
    app, 
    rag_chain.with_types(input_type=Input, output_type=Output),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
