from langchain_qdrant import Qdrant, QdrantVectorStore 
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import Optional, Literal
import uuid
from qdrant_client import QdrantClient

def get_existing_qdrant_vector_store_collection_names(
        docker_container_port: str="6333"
    ):

    client = QdrantClient(f"http://localhost:{docker_container_port}")
    collection_names = []
    for collection in client.get_collections().collections:
        collection_names.append(collection.name)        
    
    return collection_names
    



def delete_existing_qdrant_vector_store_collections(
        docker_container_port: str="6333",
        collection_names: Optional[list[str]]=None        
    ):

    client = QdrantClient(f"http://localhost:{docker_container_port}")
    if not collection_names:
        for collection in client.get_collections().collections:
            client.delete_collection(collection.name)
            # print(f"Collection {collection.name} deleted.")
    else:
        for collection_name in collection_names:
            client.delete_collection(collection_name)
            # print(f"Collection {collection_name} deleted.")
    


def get_qdrant_vector_store_from_existing_collection(
        collection_name: str,
        embedding: Embeddings,
        *,
        in_memory: bool=False,
        docker_container_port: str="6333",
        **kwargs
    ):
    assert in_memory == False, "In-memory option is not supported. Use Dcoker container instead."
    assert collection_name in get_existing_qdrant_vector_store_collection_names(docker_container_port=docker_container_port), "Collection does not exist."
    
    location = f"http://localhost:{docker_container_port}"
        
    qdrant_vector_db = Qdrant.from_existing_collection(
        collection_name=collection_name,
        location = location,
        embedding=embedding,
        **kwargs
    )    
    return qdrant_vector_db

        



def get_qdrant_vector_store_from_documents(
        documents: Document,
        embedding: Embeddings,
        collection_name: str,
        *,
        in_memory: bool=False,
        docker_container_port: str="6333",
        if_exists: Literal['replace', 'append', 'skip']='skip',
        **kwargs
    ):

    """
    Example:
        .. code-block:: python

        vector_store = get_qdrant_vector_store_from_documents(
            documents=documents[:2],
            embedding=OpenAIEmbeddings(model='text-embedding-3-small'),   
            collection_name = 'sample_collection_1'
        )
    """
    if in_memory:
        location = ":memory:"
    else:
        location = f"http://localhost:{docker_container_port}"


    if not in_memory and collection_name in get_existing_qdrant_vector_store_collection_names(docker_container_port):
        if if_exists == 'skip':
            print(f"[WARGNING] Collection {collection_name} already exists. You have chosen to skip adding the documents you provided.")

            qdrant_vector_db = Qdrant.from_existing_collection(
                collection_name=collection_name,
                location = location,
                embedding=embedding,
                **kwargs
            )    
            return qdrant_vector_db            
        
        elif if_exists == 'replace':
            delete_existing_qdrant_vector_store_collections(docker_container_port, [collection_name])
            print(f"[WARGNING] Collection {collection_name} already exists. You have chosen to replace the collection with the documents you provided.")

            qdrant_vector_db = Qdrant.from_documents(
                documents=documents,
                embedding=embedding,
                location=location,
                collection_name=collection_name,
                **kwargs
            )
            return qdrant_vector_db
        
        elif if_exists == 'append':
            print(f"[WARGNING] Collection {collection_name} already exists. You have chosen to append it with the documents you provided.")
    

    qdrant_vector_db = Qdrant.from_documents(
        documents=documents,
        embedding=embedding,
        location=location,
        collection_name=collection_name,
        **kwargs
    )    

    return qdrant_vector_db


def get_qdrant_retriever_from_documents(
        documents: Document,
        embedding: Embeddings,
        collection_name: str,
        *,
        in_memory: bool=False,
        docker_container_port: str="6333",
        if_exists: Literal['replace', 'append', 'skip']='skip',
        **kwargs        
):
    return get_qdrant_vector_store_from_documents(
        documents = documents,
        embedding = embedding,
        collection_name = collection_name,
        in_memory= in_memory,
        docker_container_port = docker_container_port,
        if_exists = if_exists,
        **kwargs        
    ).as_retriever()