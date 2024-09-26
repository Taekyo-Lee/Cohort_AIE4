import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import chainlit as cl

# LangSmith
os.environ["LANGSMITH_PROJECT"] = "AIE4_MIDTERM_DEPLOYMENT"

# cache
from langchain.globals import set_llm_cache, get_llm_cache 
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# enviorment variables
from dotenv import load_dotenv
load_dotenv()

# custom libraries
from modules.vector_stores import (
    get_existing_qdrant_vector_store_collection_names,
    get_qdrant_retriever_from_documents
)
from modules.rag_chains import get_configurable_rag_chain

# 🤗 Preprocesing
def get_configurable_rag():
    
    LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    MODELS = ['snowflake_recursive_finetuned', 'snowflake_semantic_finetuned', 'mpnet_recursive_finetuned', 'mpnet_semantic_finetuned']
    CHUNK_PATH = [
        os.path.join('documents/e2e_rags', 'chunks_recursive: size 1000 overlap 0.2.pkl'), 
        os.path.join('documents/e2e_rags', 'chunks_semantic: buffer 1 threshold 90-snowflake.pkl'), 
        os.path.join('documents/e2e_rags', 'chunks_recursive: size 1000 overlap 0.2.pkl'), 
        os.path.join('documents/e2e_rags', 'chunks_semantic: buffer 1 threshold 90-mpnet.pkl')
    ]
    CHUNKS = {}
    for model_name, chunk_path in zip(MODELS, CHUNK_PATH):
        with open(chunk_path, 'rb') as f:
            CHUNKS[model_name] = pickle.load(f)

    print("Loading embeddings...")
    EMBEDDINGS = {
    'snowflake_recursive_finetuned' : HuggingFaceEmbeddings(model_name='jet-taekyo/snowflake_finetuned_recursive'),
        'snowflake_semantic_finetuned' : HuggingFaceEmbeddings(model_name='jet-taekyo/snowflake_finetuned_semantic'),
        'mpnet_recursive_finetuned' : HuggingFaceEmbeddings(model_name='jet-taekyo/mpnet_finetuned_recursive'),
        'mpnet_semantic_finetuned' : HuggingFaceEmbeddings(model_name='jet-taekyo/mpnet_finetuned_semantic')
    }

    print("Loading retrievers...")
    RETREIVERS = {
        model_name: get_qdrant_retriever_from_documents(
            documents = chunk,
            embedding = EMBEDDINGS[model_name],
            collection_name = model_name,
            in_memory=True,
            if_exists = 'skip'
        )
        for model_name, chunk in CHUNKS.items()
    }

    print("Loading configurable RAG...")
    RAG = get_configurable_rag_chain(
        default_retriever = RETREIVERS['snowflake_recursive_finetuned'], 
        default_llm = LLM,
        default_rag_name = 'RAG'
    )

    return RETREIVERS, RAG




# 🤗 Chainlit
@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(content="Building the RAG model. Please wait...")
    await msg.send()


    retrievers, rag = get_configurable_rag()


    msg.content = (
        "Please choose one of the following models for your retriever:\n"
        "1. snowflake_recursive_finetuned\n"
        "2. snowflake_semantic_finetuned\n"
        "3. mpnet_recursive_finetuned\n"
        "4. mpnet_semantic_finetuned\n"
        "Type the number of your choice."
    )
    await msg.update()




    cl.user_session.set("retrievers", retrievers)
    cl.user_session.set("rag", rag)



@cl.on_message
async def main(message):

    retrievers = cl.user_session.get("retrievers")
    rag = cl.user_session.get("rag")


    # Check if the user has already selected a model
    selected_model = cl.user_session.get("selected_model")

    if not selected_model:
        # Map user input to model name
        model_options = {
            "1": "snowflake_recursive_finetuned",
            "2": "snowflake_semantic_finetuned",
            "3": "mpnet_recursive_finetuned",
            "4": "mpnet_semantic_finetuned"
        }

        # Check if the user input is valid
        if message.content.strip() in model_options:
            selected_model = model_options[message.content.strip()]
            cl.user_session.set("selected_model", selected_model)
            await cl.Message(content=f"Model {selected_model} selected. You can now ask questions!").send()
        else:
            await cl.Message(content="Invalid choice. Please enter a number between 1 and 4.").send()
        return




    config = {
        'configurable':{
            'retriever': retrievers[selected_model],
            'rag_name': f'RAG-{selected_model}',
        }
    }

    msg = cl.Message(content="")
    async for stream_resp in rag.astream({'question':message.content}, config=config):
        await msg.stream_token(stream_resp.content)

    await msg.send()

