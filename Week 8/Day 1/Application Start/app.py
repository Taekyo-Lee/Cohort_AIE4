### Import Section ###
import os
import re
import chainlit as cl
from langchain.storage import LocalFileStore 
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chainlit.types import AskFileResponse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import StrOutputParser
from langchain_core.documents import Document
from typing import cast
from dotenv import load_dotenv
import tempfile

### Emvironment Variables ###
load_dotenv('.env')

### Global Section ###
VECTOR_STORE_CACHE = LocalFileStore(root_path = "VECTOR_STORE_CACHE")
E2E_CACHE = LocalFileStore(root_path = "E2E_CACHE")

#😉 helper functions
def clean_text(text: str) -> str:    
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def caching_rag_respnse(question: str, answer:str):    
    E2E_CACHE.mset( [(clean_text(question), answer.encode('utf-8'))]  )

def load_cached_response(input) :
    question = clean_text(input['question']) 
    cached_answer = E2E_CACHE.mget([question])[0]
    return cached_answer.decode('utf-8') if cached_answer else False


#😉 prompt
RAG_SYSTEM_MSG_TEMPLATE = """\
You are a helpful assistant that uses the provided context to answer questions. If Context does not coantain any information to answer Question, just say "I don't know".

Question:
{question}
Context:
{context}
"""
RAG_PROMPT = ChatPromptTemplate([('human', RAG_SYSTEM_MSG_TEMPLATE)])


#😉 retriever
async def get_retriever(file: AskFileResponse):

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf') as temp_file:
        temp_file_path = temp_file.name
    with open(temp_file_path, 'wb') as f:
        f.write(file.content)
    documents = PyMuPDFLoader(temp_file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = await text_splitter.atransform_documents(documents)


    client = QdrantClient(":memory:")
    core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings = core_embeddings, 
        document_embedding_cache = VECTOR_STORE_CACHE, 
        namespace=core_embeddings.model 
    )

    
    collection_name = f"pdf_to_parse_{clean_text(file.name)}"
    if collection_name not in (x.name for x in client.get_collections().collections):    
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )    
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=cached_embedder
            )    
        vectorstore.add_documents(chunks)
        already_exist = False
    else:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=cached_embedder
            )    
        already_exist = True
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    return retriever, already_exist


def get_rag(retriever):
    chat_model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    rag_chain =  RunnableParallel(
        context = retriever,
        question = lambda x: x 
    )| RAG_PROMPT | chat_model | StrOutputParser()
    rag_chain = rag_chain.with_config({'run_name':'RAG'})

    return rag_chain





### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    """ SESSION SPECIFIC CODE HERE """
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Hello!! I'm Jet! Please upload a Pdf File file to begin!",
            accept=["application/pdf"],  
            max_size_mb=10,
            timeout=180,
        ).send()


    file = files[0] 
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_human_feedback=True)
    await msg.send()

    
    # get rag chain
    retriever, already_exist = await get_retriever(file)
    # retriever, already_exist = await get_retriever(file.name.split('pdf')[0], chunks)
    rag_chain = get_rag(retriever)

    # Let the user know that the system is ready
    if not already_exist:
        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    else:
        msg.content = f"VectorStore already exist. You can now ask questions!" 
    await msg.update()

    cl.user_session.set("chain", rag_chain)




### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """
    rename_dict = {"Assistant": "Jet"}
    return rename_dict.get(orig_author, orig_author)


### On Message Section ###
@cl.on_message
async def main(message):
    """
    MESSAGE CODE HERE
    """

    cached_answer = load_cached_response({'question':message.content})
    if cached_answer:
        msg = cl.Message(content=cached_answer)
        await msg.send()
    else:
        chain = cast(Runnable, cl.user_session.get("chain")) 

        msg = cl.Message(content="")
        async for stream_resp in chain.astream(message.content):
            await msg.stream_token(stream_resp)

        caching_rag_respnse(question=message.content, answer=msg.content)

        await msg.send()


